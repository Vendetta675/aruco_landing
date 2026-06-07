#!/usr/bin/env python3
"""
ArUco Precision-Landing Node  —  tuned for 2 m × 2 m marker
=============================================================

All parameters are calibrated for:
- ArUco DICT_4X4_50, marker ID 0, physical size 2.0 m × 2.0 m
- Downward-facing gimbal camera, 640×480, fx=fy≈205.5

tvec convention (estimatePoseSingleMarkers, nadir camera):
tvec[0]  cam_x  right=+   
tvec[1]  cam_y  fwd/down  
tvec[2]  cam_z  depth     =  altitude above marker  (scales with marker_size)

Stage-5 sub-states
------------------
SEARCHING    Gentle expanding-square drift (rate-limited, no jerks).
             Altitude held constant.
STABILISING  Marker just appeared — hold position, wait N clean frames
             before starting descent (prevents jerk on noisy first frame).
TRACKING     Calculates global marker position with YAW COMPENSATION, coasts toward it, descends once lateral < threshold.
BLIND_DESCENT cam_z < BLIND_ALT_THRESHOLD — last good XY frozen,
             descend straight down.
"""

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import time
import math


# =========================================================================== #
# TVEC VALIDATOR                                                               #
# =========================================================================== #

class TvecValidator:
    """Rejects physically implausible tvec readings."""
    MAX_LATERAL_M = 30.0   # 2 m marker → errors up to ~15 m are valid
    MAX_CAM_Z_M   = 40.0
    MAX_JUMP_M    = 8.0    # max frame-to-frame change

    def __init__(self):
        self._prev = None

    def reset(self):
        self._prev = None

    def validate(self, tvec):
        x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
        if z <= 0.1 or z > self.MAX_CAM_Z_M:
            return False
        if abs(x) > self.MAX_LATERAL_M or abs(y) > self.MAX_LATERAL_M:
            return False
        if self._prev is not None:
            jump = np.linalg.norm(np.array([x, y, z]) - np.array(self._prev))
            if jump > self.MAX_JUMP_M:
                return False
        self._prev = [x, y, z]
        return True


# =========================================================================== #
# GENTLE EXPANDING-SQUARE SEARCH                                               #
# =========================================================================== #

class GentleExpandingSearch:
    """
    Outward expanding-square spiral.
    Setpoint changes are rate-limited → smooth drift, never jerky.
    Altitude held constant throughout.
    """
    SEARCH_STEP = 1.0   # m — square expands by this each revolution
    MAX_RADIUS  = 10.0   # m — give up beyond this
    WP_RADIUS   = 0.4    # m — waypoint acceptance radius
    WP_TIMEOUT  = 15.0   # s — hard per-waypoint time limit
    MAX_SP_RATE = 0.20   # m/s — rate-limit on setpoint motion

    def __init__(self):
        self.active = False
        self.origin_x = self.origin_y = self.altitude = 0.0
        self._waypoints = []; self._wp_idx = 0; self._wp_time = 0.0
        self._smooth_x = self._smooth_y = 0.0; self._last_t = None

    def start(self, ox, oy, alt):
        self.active = True
        self.origin_x = ox; self.origin_y = oy; self.altitude = alt
        self._smooth_x = ox; self._smooth_y = oy
        self._last_t = time.time()
        self._waypoints = self._spiral(ox, oy)
        self._wp_idx = 0; self._wp_time = time.time()

    def reset(self):
        self.active = False

    def _spiral(self, ox, oy):
        wps = []; x, y = ox, oy; step = self.SEARCH_STEP
        leg = 1; dirs = [(1,0),(0,1),(-1,0),(0,-1)]; di = 0
        while True:
            for _ in range(2):
                dx, dy = dirs[di % 4]; di += 1
                for _ in range(leg):
                    x += dx * step; y += dy * step
                    if math.sqrt((x-ox)**2+(y-oy)**2) > self.MAX_RADIUS:
                        return wps
                    wps.append((x, y))
            leg += 1

    def update(self, drone_x, drone_y):
        """Returns (sp_x, sp_y, exhausted)."""
        if not self.active or not self._waypoints:
            return self.origin_x, self.origin_y, True
        now = time.time()
        dt = (now - self._last_t) if self._last_t else 0.05
        self._last_t = now
        if self._wp_idx < len(self._waypoints):
            wx, wy = self._waypoints[self._wp_idx]
            if (math.sqrt((drone_x-wx)**2+(drone_y-wy)**2) < self.WP_RADIUS
                    or now - self._wp_time > self.WP_TIMEOUT):
                self._wp_idx += 1; self._wp_time = now
        if self._wp_idx >= len(self._waypoints):
            return self._smooth_x, self._smooth_y, True
        wx, wy = self._waypoints[self._wp_idx]
        ex = wx - self._smooth_x; ey = wy - self._smooth_y
        d = math.sqrt(ex**2 + ey**2)
        if d > 1e-4:
            mv = min(self.MAX_SP_RATE * dt, d)
            self._smooth_x += ex/d * mv; self._smooth_y += ey/d * mv
        return self._smooth_x, self._smooth_y, False


# =========================================================================== #
# MAIN NODE                                                                    #
# =========================================================================== #

class TakeoffPIDLand(Node):

    SS_SEARCHING   = 'SEARCHING'
    SS_STABILISING = 'STABILISING'
    SS_TRACKING    = 'TRACKING'
    SS_BLIND       = 'BLIND_DESCENT'

    # ── MARKER ────────────────────────────────────────────────────────────
    MARKER_SIZE = 2.0           # metres

    # ── LANDING GEOMETRY ──────────────────────────────────────────────────
    LANDING_ALTITUDE    = 1.0   # m above marker → trigger LAND mode
    LANDING_DEADBAND    = 0.50  # m lateral tolerance for LAND

    # ── BLIND DESCENT ─────────────────────────────────────────────────────
    BLIND_ALT_THRESHOLD = 1.0   # m
    BLIND_DESCENT_RATE  = 0.03  # m lowered per control cycle

    # ── SETPOINT CLAMPS ───────────────────────────────────────────────────
    MAX_SP_DIST_XY = 1.0        # m — max setpoint offset from current pos XY
    MAX_SP_DIST_Z  = 0.6        # m — max setpoint offset from current pos Z

    # ── DETECTION ─────────────────────────────────────────────────────────
    LOST_FRAME_THRESHOLD = 6    # consecutive missed frames → SEARCHING
    STABILISE_FRAMES     = 8    # consecutive valid frames → leave STABILISING

    # ── DESCENT GATE ──────────────────────────────────────────────────────
    CENTRE_THRESHOLD = 0.60     # m in cam frame
    DESCENT_STEP_M   = 0.02     # m per control cycle — gradual, not aggressive

    def __init__(self):
        super().__init__('auto_takeoff')

        self.state = State()
        self.x_pos = self.y_pos = self.z_pos = 0.0
        self.sp_x  = self.sp_y = self.sp_z   = 0.0

        self.stage             = 0
        self.altitude_received = False
        self.sub_state         = self.SS_SEARCHING

        self._marker_visible = False
        self._last_tvec      = None
        self.lost_frames     = 0
        self._stab_frames    = 0
        self._blind_sp_x     = None
        self._blind_sp_y     = None
        
        # Position & Yaw Memory Variables
        self.last_marker_global_x = None
        self.last_marker_global_y = None
        self.yaw                  = 0.0  # Added for coordinate rotation

        self.validator = TvecValidator()
        self.search    = GentleExpandingSearch()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.bridge = CvBridge()

        # OpenCV ArUco — DICT_4X4_50, marker ID 0
        self.aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params   = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params)

        self.camera_matrix = np.array([
            [205.4696273803711, 0.0,               320.0],
            [0.0,               205.4696559906006, 240.0],
            [0.0,               0.0,                 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.create_subscription(State,
            '/mavros/state', self.state_cb, 10)
        self.create_subscription(PoseStamped,
            '/mavros/local_position/pose', self.pos_cb, qos)
        self.create_subscription(Image,
            '/camera_image', self.image_callback, 10)

        self.pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)

        self.arming_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client    = self.create_client(SetMode,     '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL,  '/mavros/cmd/takeoff')

        self.timer    = self.create_timer(0.05, self.control_loop)
        self.sp_timer = self.create_timer(0.05, self._publish_setpoint)

        self.get_logger().info(
            f'TakeoffPIDLand started | '
            f'MARKER_SIZE={self.MARKER_SIZE}m | '
            f'BLIND_THRESH={self.BLIND_ALT_THRESHOLD}m | '
            f'CENTRE_THRESH={self.CENTRE_THRESHOLD}m')

    # ================================================================== #
    # CALLBACKS                                                           #
    # ================================================================== #

    def state_cb(self, msg):
        self.state = msg

    def pos_cb(self, msg):
        self.x_pos = msg.pose.position.x
        self.y_pos = msg.pose.position.y
        self.z_pos = msg.pose.position.z
        
        # Convert MAVROS quaternion to Euler Yaw (radians)
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.altitude_received = True

    # ================================================================== #
    # IMAGE CALLBACK                                                      #
    # ================================================================== #

    def image_callback(self, msg):
        if self.stage != 5:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        detected = False
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid == 0:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], self.MARKER_SIZE,
                        self.camera_matrix, self.dist_coeffs)
                    tvec = tvecs[0][0]
                    if self.validator.validate(tvec):
                        self._on_marker_detected(tvec)
                        self.lost_frames = 0
                        detected = True
                        break

        if not detected:
            self.lost_frames += 1
            if self.lost_frames >= self.LOST_FRAME_THRESHOLD:
                self._on_marker_lost()

    # ================================================================== #
    # MARKER DETECTED                                                     #
    # ================================================================== #

    def _on_marker_detected(self, tvec):
        self._last_tvec = np.array(tvec, dtype=float).copy()

        # Blind descent: ignore new detections, keep XY frozen
        if self.sub_state == self.SS_BLIND:
            return

        # First frame after loss → stabilise before doing anything
        if not self._marker_visible:
            self._marker_visible = True
            self._stab_frames    = 0
            self.search.reset()
            self.sub_state = self.SS_STABILISING
            self.get_logger().info(
                f'[STABILISING] Marker acquired | '
                f'cam({tvec[0]:+.2f}, {tvec[1]:+.2f}, {tvec[2]:.2f})')

        if self.sub_state == self.SS_STABILISING:
            self._stab_frames += 1
            if self._stab_frames >= self.STABILISE_FRAMES:
                self.sub_state = self.SS_TRACKING
                self.get_logger().info('[TRACKING] Stable — beginning descent.')
            return   # do NOT move setpoint during stabilisation

        if self.sub_state == self.SS_TRACKING:
            self._run_tracking(tvec)

    # ------------------------------------------------------------------ #
    # GLOBAL TRACKING (WITH YAW ROTATION)                                 #
    # ------------------------------------------------------------------ #

    def _run_tracking(self, tvec):
        cam_x = float(tvec[0])
        cam_y = float(tvec[1])
        cam_z = float(tvec[2])
        lateral = math.sqrt(cam_x**2 + cam_y**2)

        if cam_z < self.BLIND_ALT_THRESHOLD:
            self._enter_blind_descent()
            return

        # 1. Map camera frame to Drone Body frame (Forward-Left-Up)
        # Standard nadir camera: top of image (-cam_y) is forward, right (+cam_x) is right
        body_forward = cam_y
        body_left    = -cam_x  # left is opposite of right

        # 2. Rotate body frame error to Global ENU map frame using drone's yaw
        # This completely negates spiraling/drifting no matter the drone's heading.
        global_err_x = (body_forward * math.cos(self.yaw)) - (body_left * math.sin(self.yaw))
        global_err_y = (body_forward * math.sin(self.yaw)) + (body_left * math.cos(self.yaw))

        marker_global_x = self.x_pos + global_err_x
        marker_global_y = self.y_pos + global_err_y

        self.last_marker_global_x = marker_global_x
        self.last_marker_global_y = marker_global_y

        # 3. SMOOTHLY MOVE SETPOINT TOWARDS MARKER
        P_gain = 0.6 
        target_x = self.x_pos + P_gain * (marker_global_x + self.x_pos)
        target_y = self.y_pos + P_gain * (marker_global_y + self.y_pos)

        # CLAMP THE SETPOINT: Prevents aggressive pitch and camera FOV loss
        self.sp_x = float(np.clip(target_x, self.x_pos - self.MAX_SP_DIST_XY, self.x_pos + self.MAX_SP_DIST_XY))
        self.sp_y = float(np.clip(target_y, self.y_pos - self.MAX_SP_DIST_XY, self.y_pos + self.MAX_SP_DIST_XY))

        # Altitude: only descend once centred
        if lateral < self.CENTRE_THRESHOLD:
            self.sp_z = float(np.clip(
                self.sp_z - self.DESCENT_STEP_M,
                self.z_pos - self.MAX_SP_DIST_Z,
                self.z_pos + self.MAX_SP_DIST_Z))

        self.get_logger().info(
            f'[TRACKING] '
            f'lat:{lateral:.3f}m | yaw:{math.degrees(self.yaw):.1f}° | '
            f'sp({self.sp_x:.2f},{self.sp_y:.2f},{self.sp_z:.2f})')

    # ================================================================== #
    # BLIND DESCENT                                                       #
    # ================================================================== #

    def _enter_blind_descent(self):
        self.sub_state   = self.SS_BLIND
        self._blind_sp_x = self.sp_x
        self._blind_sp_y = self.sp_y
        self.get_logger().warn(
            f'[BLIND_DESCENT] Freezing XY at '
            f'({self._blind_sp_x:.2f}, {self._blind_sp_y:.2f}) | '
            f'drone_z={self.z_pos:.2f}m')

    def _update_blind_descent(self):
        if self._blind_sp_x is not None:
            self.sp_x = self._blind_sp_x
            self.sp_y = self._blind_sp_y
        self.sp_z = max(0.0, self.sp_z - self.BLIND_DESCENT_RATE)

    # ================================================================== #
    # MARKER LOST - COASTING / SEARCHING                                  #
    # ================================================================== #

    def _on_marker_lost(self):
        if self._marker_visible:
            self._marker_visible = False
            self._stab_frames    = 0
            lx = self.last_marker_global_x if self.last_marker_global_x is not None else 0.0
            ly = self.last_marker_global_y if self.last_marker_global_y is not None else 0.0
            self.get_logger().warn(
                f'[LOST] Marker lost! Coasting to pos({lx:.2f},{ly:.2f})')

        # Already very close — keep descending blind
        if self.sub_state == self.SS_BLIND:
            return

        # NEW LOGIC: If we are already searching, DO NOT snap back to coasting!
        if self.sub_state == self.SS_SEARCHING:
            sp_x, sp_y, exhausted = self.search.update(self.x_pos, self.y_pos)
            if exhausted:
                self.get_logger().error('[SEARCHING] Exhausted — holding position.')
            self.sp_x = float(sp_x)
            self.sp_y = float(sp_y)
            return

        # Coasting logic to last known position
        if self.last_marker_global_x is not None and self.last_marker_global_y is not None:
            dist_to_last = math.sqrt((self.x_pos - self.last_marker_global_x)**2 + (self.y_pos - self.last_marker_global_y)**2)
            
            # If we reached the spot and STILL don't see it, trigger expanding search
            if dist_to_last < 0.3:
                self.sub_state = self.SS_SEARCHING
                self.search.start(self.x_pos, self.y_pos, self.z_pos)
                
                sp_x, sp_y, _ = self.search.update(self.x_pos, self.y_pos)
                self.sp_x = float(sp_x)
                self.sp_y = float(sp_y)
            else:
                # Smoothly glide to the last known position (clamped to prevent pitching)
                P_gain = 0.6
                target_x = self.x_pos + P_gain * (self.last_marker_global_x - self.x_pos)
                target_y = self.y_pos + P_gain * (self.last_marker_global_y - self.y_pos)
                
                self.sp_x = float(np.clip(target_x, self.x_pos - self.MAX_SP_DIST_XY, self.x_pos + self.MAX_SP_DIST_XY))
                self.sp_y = float(np.clip(target_y, self.y_pos - self.MAX_SP_DIST_XY, self.y_pos + self.MAX_SP_DIST_XY))
        
        # Fallback if marker was never seen
        else:
            self.sub_state = self.SS_SEARCHING
            self.search.start(self.x_pos, self.y_pos, self.z_pos)

            sp_x, sp_y, exhausted = self.search.update(self.x_pos, self.y_pos)
            if exhausted:
                self.get_logger().error('[SEARCHING] Exhausted — holding position.')
            self.sp_x = float(sp_x)
            self.sp_y = float(sp_y)

    # ================================================================== #
    # SETPOINT PUBLISHER                                                  #
    # ================================================================== #

    def _publish_setpoint(self):
        if self.stage < 4:
            return
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = self.sp_x
        msg.pose.position.y = self.sp_y
        msg.pose.position.z = self.sp_z
        msg.pose.orientation.w = 1.0
        self.pos_pub.publish(msg)

    # ================================================================== #
    # CONTROL LOOP                                                        #
    # ================================================================== #

    def control_loop(self):
        if not self.state.connected:
            return

        if self.stage == 0:
            if self.state.mode != 'GUIDED':
                req = SetMode.Request(); req.custom_mode = 'GUIDED'
                self.mode_client.call_async(req)
            else:
                self.stage = 1

        elif self.stage == 1:
            if not self.state.armed:
                req = CommandBool.Request(); req.value = True
                self.arming_client.call_async(req)
            else:
                self.stage = 2

        elif self.stage == 2:
            req = CommandTOL.Request(); req.altitude = 10.0
            self.takeoff_client.call_async(req)
            self.stage = 3
            self.get_logger().info('Takeoff command sent.')

        elif self.stage == 3:
            if self.z_pos >= 2.5:
                self.sp_x = self.x_pos
                self.sp_y = self.y_pos
                self.sp_z = self.z_pos
                self.stage = 4
                self.get_logger().info(f'Altitude reached ({self.z_pos:.2f}m).')

        elif self.stage == 4:
            self.stage = 5
            self.search.start(self.x_pos, self.y_pos, self.z_pos)
            self.get_logger().info('Landing stage started.')

        elif self.stage == 5:
            if self.sub_state == self.SS_BLIND:
                self._update_blind_descent()

            # Normal land: centred on marker at or below 3m altitude
            lateral_error = math.sqrt(self._last_tvec[0]**2 + self._last_tvec[1]**2) if self._last_tvec is not None else 99.0
            
            normal_land = (
                self._marker_visible
                and self.sub_state == self.SS_TRACKING
                and self._last_tvec is not None
                and lateral_error < self.LANDING_DEADBAND
                and self._last_tvec[2] <= (self.LANDING_ALTITUDE + 0.2))

            # Blind land: sp_z has reached floor
            blind_land = (
                self.sub_state == self.SS_BLIND
                and self.sp_z <= self.LANDING_ALTITUDE)

            if normal_land or blind_land:
                reason = 'centred on marker' if normal_land else 'blind descent complete'
                self.get_logger().info(f'Triggering LAND — {reason}.')
                self.stage = 6

        elif self.stage == 6:
            if self.state.mode != 'LAND':
                req = SetMode.Request(); req.custom_mode = 'LAND'
                self.mode_client.call_async(req)
                self.get_logger().info('LAND mode commanded.')
            else:
                self.get_logger().info('LAND mode active.')
                self.timer.cancel(); self.sp_timer.cancel()


# =========================================================================== #
# MAIN                                                                         #
# =========================================================================== #

def main(args=None):
    rclpy.init(args=args)
    node = TakeoffPIDLand()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()