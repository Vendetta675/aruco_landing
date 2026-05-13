#!/usr/bin/env python3

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
# PID                                                                          #
# =========================================================================== #

class PID:

    def __init__(self, kp, ki, kd, limit):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self.limit = limit
        self.integral   = 0.0
        self.prev_error = 0.0
        self.prev_time  = None

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0
        self.prev_time  = None

    def update(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            # On first call return a proportional-only output so we
            # don't waste the first frame with zero output.
            return float(np.clip(self.kp * error, -self.limit, self.limit))
        dt = now - self.prev_time
        self.prev_time = now
        if dt <= 0.0:
            return float(np.clip(self.kp * error, -self.limit, self.limit))
        self.integral = np.clip(
            self.integral + error * dt, -self.limit / self.kp, self.limit / self.kp)
        derivative      = (error - self.prev_error) / dt
        self.prev_error = error
        out = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        return float(np.clip(out, -self.limit, self.limit))


# =========================================================================== #
# TVEC VALIDATOR                                                               #
# =========================================================================== #

class TvecValidator:

    MAX_LATERAL_M = 15.0
    MAX_CAM_Z_M   = 25.0
    MAX_JUMP_M    = 4.0

    def __init__(self):
        self._prev = None

    def reset(self):
        self._prev = None

    def validate(self, tvec):
        x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
        if z <= 0.05 or z > self.MAX_CAM_Z_M:
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
# SPIRAL SEARCH                                                                #
#                                                                              #
# When the marker is lost we execute an outward square spiral from the        #
# last-known position. Each leg grows by STEP_M every two legs so the drone   #
# covers progressively more area.  Once the marker is re-acquired the search  #
# is cancelled.                                                                #
# =========================================================================== #

class SpiralSearch:

    # metres per spiral leg (grows every 2 legs)
    STEP_M       = 0.6
    # time to travel each leg before moving to the next waypoint (seconds)
    LEG_DWELL    = 2.5
    # maximum spiral radius before giving up
    MAX_RADIUS_M = 3.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.active        = False
        self.origin_x      = 0.0
        self.origin_y      = 0.0
        self.origin_z      = 0.0
        self.wp_x          = 0.0
        self.wp_y          = 0.0
        self.leg_index     = 0
        self.leg_start     = 0.0
        self.current_step  = self.STEP_M
        # Directions: right, forward, left, back (world X/Y)
        self._dirs = [(1,0), (0,1), (-1,0), (0,-1)]

    def start(self, origin_x, origin_y, origin_z):
        self.reset()
        self.active    = True
        self.origin_x  = origin_x
        self.origin_y  = origin_y
        self.origin_z  = origin_z
        self.wp_x      = origin_x
        self.wp_y      = origin_y
        self.leg_start = time.time()
        self.get_next_waypoint()   # set first wp

    def get_next_waypoint(self):
        dir_idx  = self.leg_index % 4
        leg_mult = (self.leg_index // 2) + 1
        step     = self.STEP_M * leg_mult
        dx = self._dirs[dir_idx][0] * step
        dy = self._dirs[dir_idx][1] * step
        self.wp_x += dx
        self.wp_y += dy
        self.leg_index += 1
        self.leg_start = time.time()
        return self.wp_x, self.wp_y

    def update(self, drone_x, drone_y):

        dist = math.sqrt(
        (drone_x - self.wp_x)**2 +
        (drone_y - self.wp_y)**2)
        elapsed = time.time() - self.leg_start
        if dist < 0.15 or elapsed > 4.0:
            self.get_next_waypoint()
        return self.wp_x, self.wp_y, False
        self.sp_x = target_x
        self.sp_y = target_y
        self.sp_z = self.spiral.origin_z


# =========================================================================== #
# MAIN NODE                                                                    #
# =========================================================================== #

class TakeoffPIDLand(Node):

    # ── Landing geometry ─────────────────────────────────────────────────
    LANDING_ALTITUDE = 0.5    # m above marker — trigger LAND below this
    LANDING_DEADBAND = 0.08   # m  lateral/altitude tolerance to trigger LAND

    # ── Setpoint limits ───────────────────────────────────────────────────
    # These limit how far the SETPOINT may be from current POSITION.
    # Larger = more aggressive response; smaller = safer but sluggish.
    MAX_SP_DIST_XY   = 0.60   # m  — setpoint may lead drone by this much
    MAX_SP_DIST_Z    = 0.30   # m

    # ── Detection ────────────────────────────────────────────────────────
    LOST_FRAME_THRESHOLD = 6

    # ── Tracking mode IDs ─────────────────────────────────────────────────
    TRACK_ARUCO  = 0
    TRACK_SPIRAL = 1
    TRACK_HOLD   = 2

    def __init__(self):

        super().__init__('auto_takeoff')

        self.state = State()

        self.x_pos = self.y_pos = self.z_pos = 0.0
        self.sp_x  = self.sp_y = self.sp_z  = 0.0

        self.stage             = 0
        self.altitude_received = False
        self.tracking_mode     = self.TRACK_HOLD

        self._marker_visible = False
        self._last_tvec      = None
        self.lost_frames     = 0

        self.last_detection_time = time.time()

        self.validator = TvecValidator()
        self.spiral    = SpiralSearch()

        # ── ArUco landing PIDs ────────────────────────────────────────────
        #
        # DESIGN RATIONALE
        # ─────────────────
        # We want the drone to move QUICKLY toward the marker centre so
        # it converges before losing sight of it.  Previous versions used
        # kp=0.4–0.8; this caused slow convergence → orbit → marker loss.
        #
        # kp=1.2 means: 1 m lateral error → 1.2 m/cycle command (clamped
        # to MAX_SP_DIST_XY=0.60).  At 20 Hz the drone covers ~0.60 m per
        # 50 ms step.  The flight controller will smoothly execute this.
        #
        # kd=0.15 damps oscillation as the drone approaches centre.
        # ki=0.008 removes steady-state offset (e.g. constant wind bias).
        #
        # Z: kp=0.9 so a 2 m altitude error → 0.9*2=1.8 (clamped 0.30)
        # → 0.30 m/step descent.  Fast but safe.

        self.pid_x = PID(kp=1.2, ki=0.008, kd=0.15, limit=self.MAX_SP_DIST_XY)
        self.pid_y = PID(kp=1.2, ki=0.008, kd=0.15, limit=self.MAX_SP_DIST_XY)
        self.pid_z = PID(kp=0.9, ki=0.010, kd=0.05, limit=self.MAX_SP_DIST_Z)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.bridge       = CvBridge()
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.camera_matrix = np.array([
            [205.4696273803711, 0.0,               320.0],
            [0.0,               205.4696559906006, 240.0],
            [0.0,               0.0,                 1.0]
        ])
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_size = 0.2   # metres

        self.create_subscription(State,       '/mavros/state',               self.state_cb,       10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb,         qos)
        self.create_subscription(Image,       '/camera_image',               self.image_callback, 10)

        self.pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)

        self.arming_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client    = self.create_client(SetMode,     '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL,  '/mavros/cmd/takeoff')

        self.timer    = self.create_timer(0.05, self.control_loop)
        self.sp_timer = self.create_timer(0.05, self._publish_setpoint)

        self.get_logger().info('TakeoffPIDLand node started.')

    # ================================================================== #
    # CALLBACKS                                                           #
    # ================================================================== #

    def state_cb(self, msg):
        self.state = msg

    def pos_cb(self, msg):
        self.x_pos = msg.pose.position.x
        self.y_pos = msg.pose.position.y
        self.z_pos = msg.pose.position.z
        self.altitude_received = True

    # ================================================================== #
    # IMAGE CALLBACK                                                      #
    # ================================================================== #

    def image_callback(self, msg):

        if self.stage != 5:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:

            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size,
                self.camera_matrix, self.dist_coeffs)

            tvec = tvecs[0][0]

            if self.validator.validate(tvec):
                self.lost_frames         = 0
                self.last_detection_time = time.time()
                self._on_aruco_detected(tvec)
                return

        # Marker not seen this frame
        self.lost_frames += 1
        if self.lost_frames >= self.LOST_FRAME_THRESHOLD:
            self._on_marker_lost()

    # ================================================================== #
    # ARUCO DETECTED                                                      #
    # ================================================================== #

    def _on_aruco_detected(self, tvec):
        """
        Downward-facing camera → world frame:
        tvec[0]  cam_x  right=+   maps to world Y
        tvec[1]  cam_y  fwd=+     maps to world X
        tvec[2]  cam_z  depth = altitude above marker

        Control law
        ───────────
        The setpoint is computed as:

            sp_x = x_pos + pid_x(cam_y)   ← move forward/back to zero cam_y
            sp_y = y_pos + pid_y(cam_x)   ← move left/right  to zero cam_x
            sp_z = z_pos - pid_z(cam_z - LANDING_ALTITUDE)

        We accumulate from the CURRENT SETPOINT (not pos) so the
        setpoint leads the drone.  The safety clamp keeps sp within
        MAX_SP_DIST_XY of current position.

        KEY INSIGHT: We do NOT reset the setpoint to pos on
        re-acquisition.  We let it accumulate so the drone keeps
        moving toward the marker even if detection flickers.
        """
        was_lost = not self._marker_visible

        self._last_tvec      = np.array(tvec).copy()
        self._marker_visible = True

        if self.spiral.active:
            self.spiral.reset()
            self.get_logger().info('[SPIRAL] Cancelled — marker re-acquired.')

        self.tracking_mode = self.TRACK_ARUCO

        if was_lost:
            # Reset PIDs on re-acquisition to clear stale integrals
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
            self.get_logger().info('[ARUCO] Re-acquired — PIDs reset.')

        cam_x = float(tvec[0])   # lateral (right=+)
        cam_y = float(tvec[1])   # forward (fwd=+)
        cam_z = float(tvec[2])   # altitude above marker

        # XY: error = offset of marker in camera frame
        # Positive cam_y means marker is AHEAD → move sp_x forward
        # Positive cam_x means marker is RIGHT  → move sp_y rightward
        delta_x = self.pid_x.update(cam_y)
        delta_y = self.pid_y.update(cam_x)

        # Z: descend until cam_z == LANDING_ALTITUDE
        # cam_z > LANDING_ALTITUDE → too high → negative delta_z (descend)
        altitude_error = cam_z - self.LANDING_ALTITUDE
        delta_z        = -self.pid_z.update(altitude_error)
        # Hard cap: never command more than 0.12 m descent per cycle
        delta_z = float(np.clip(delta_z, -0.12, 0.12))

        # Accumulate setpoint from PREVIOUS setpoint (not from pos)
        # so it leads the drone toward the marker.
        new_sp_x = self.sp_x + delta_x
        new_sp_y = self.sp_y + delta_y
        new_sp_z = self.sp_z + delta_z

        # Safety clamp: sp may not exceed MAX_SP_DIST from current pos
        self.sp_x = float(np.clip(new_sp_x,
            self.x_pos - self.MAX_SP_DIST_XY,
            self.x_pos + self.MAX_SP_DIST_XY))
        self.sp_y = float(np.clip(new_sp_y,
            self.y_pos - self.MAX_SP_DIST_XY,
            self.y_pos + self.MAX_SP_DIST_XY))
        self.sp_z = float(np.clip(new_sp_z,
            self.z_pos - self.MAX_SP_DIST_Z,
            self.z_pos + self.MAX_SP_DIST_Z))

        lateral_dist = math.sqrt(cam_x**2 + cam_y**2)

        self.get_logger().info(
            f'[ARUCO] '
            f'cam({cam_x:+.3f}, {cam_y:+.3f}, {cam_z:.3f}) | '
            f'lateral:{lateral_dist:.3f}m | '
            f'alt_err:{altitude_error:+.3f} | '
            f'Δ({delta_x:+.3f}, {delta_y:+.3f}, {delta_z:+.3f}) | '
            f'sp({self.sp_x:.2f}, {self.sp_y:.2f}, {self.sp_z:.2f})'
        )

    # ================================================================== #
    # MARKER LOST → SPIRAL SEARCH                                         #
    # ================================================================== #

    def _on_marker_lost(self):
        """
        Replace dead-reckoning with a spiral search.

        WHY SPIRAL INSTEAD OF DR
        ────────────────────────
        Dead-reckoning propagates a tvec estimate into world coordinates.
        This estimate is only as good as the last tvec, which is often
        noisy (the marker was partially visible, or the drone was moving).
        A bad seed sends the drone in the wrong direction, making recovery
        worse, not better.

        A spiral search makes NO assumption about where the marker is.
        It systematically covers the area around the last-known drone
        position until the marker re-enters the camera FOV.  Because the
        marker is always close (we saw it recently), a small spiral
        (radius ≤ 3 m, step 0.6 m) reliably re-acquires it.
        """
        if self._marker_visible:
            # Transition: was tracking, now lost
            self._marker_visible = False
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()

            if not self.spiral.active:
                # Start spiral from current position, hold current altitude
                self.spiral.start(self.x_pos, self.y_pos, self.z_pos)
                self.tracking_mode = self.TRACK_SPIRAL
                self.get_logger().warn(
                    f'[SPIRAL] Started from '
                    f'({self.x_pos:.2f}, {self.y_pos:.2f}, {self.z_pos:.2f})'
                )

        if not self.spiral.active:
            # Spiral was never started (lost_frames fired without prior
            # ARUCO track) — just hold position
            self.tracking_mode = self.TRACK_HOLD
            self.sp_x = self.x_pos
            self.sp_y = self.y_pos
            self.sp_z = self.z_pos
            return

        # ── Execute spiral waypoint ───────────────────────────────────
        target_x, target_y, exceeded = self.spiral.update(
            self.x_pos, self.y_pos)

        if exceeded:
            self.get_logger().error(
                '[SPIRAL] Max radius reached — HOLD.')
            self.spiral.reset()
            self.tracking_mode = self.TRACK_HOLD
            self.sp_x = self.x_pos
            self.sp_y = self.y_pos
            self.sp_z = self.z_pos
            return

        # Command setpoint directly to the spiral waypoint.
        # Clamp to MAX_SP_DIST_XY from current pos for safety.
        self.sp_x = float(np.clip(
            target_x,
            self.x_pos - self.MAX_SP_DIST_XY,
            self.x_pos + self.MAX_SP_DIST_XY))
        self.sp_y = float(np.clip(
            target_y,
            self.y_pos - self.MAX_SP_DIST_XY,
            self.y_pos + self.MAX_SP_DIST_XY))
        self.sp_z = self.spiral.origin_z   # hold altitude from spiral start

        dist_to_wp = math.sqrt(
            (target_x - self.x_pos)**2 +
            (target_y - self.y_pos)**2)

        self.get_logger().warn(
            f'[SPIRAL] leg:{self.spiral.leg_index} | '
            f'wp({target_x:.2f}, {target_y:.2f}) | '
            f'dist_to_wp:{dist_to_wp:.2f}m | '
            f'sp({self.sp_x:.2f}, {self.sp_y:.2f}, {self.sp_z:.2f})'
        )

    # ================================================================== #
    # SETPOINT PUBLISHER                                                  #
    # ================================================================== #

    def _publish_setpoint(self):

        if self.stage < 4:
            return

        msg = PoseStamped()
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.header.frame_id    = 'map'
        msg.pose.position.x    = self.sp_x
        msg.pose.position.y    = self.sp_y
        msg.pose.position.z    = self.sp_z
        msg.pose.orientation.w = 1.0

        self.pos_pub.publish(msg)

    # ================================================================== #
    # CONTROL LOOP                                                        #
    # ================================================================== #

    def control_loop(self):

        if not self.state.connected:
            return

        # Stage 0: GUIDED mode
        if self.stage == 0:
            if self.state.mode != 'GUIDED':
                req = SetMode.Request()
                req.custom_mode = 'GUIDED'
                self.mode_client.call_async(req)
            else:
                self.stage = 1

        # Stage 1: Arm
        elif self.stage == 1:
            if not self.state.armed:
                req = CommandBool.Request()
                req.value = True
                self.arming_client.call_async(req)
            else:
                self.stage = 2

        # Stage 2: Takeoff
        elif self.stage == 2:
            req          = CommandTOL.Request()
            req.altitude = 10.0
            self.takeoff_client.call_async(req)
            self.stage   = 3
            self.get_logger().info('Takeoff command sent.')

        # Stage 3: Wait for altitude
        elif self.stage == 3:
            if self.z_pos >= 2.5:
                self.sp_x = self.x_pos
                self.sp_y = self.y_pos
                self.sp_z = self.z_pos
                self.stage = 4
                self.get_logger().info(
                    f'Altitude reached ({self.z_pos:.2f}m).')

        # Stage 4: Transition to landing
        elif self.stage == 4:
            self.stage = 5
            self.get_logger().info('Landing stage started.')

        # Stage 5: ArUco PID landing
        elif self.stage == 5:

            status_map = {
                self.TRACK_ARUCO:  'ARUCO',
                self.TRACK_SPIRAL: 'SPIRAL',
                self.TRACK_HOLD:   'HOLD',
            }
            self.get_logger().info(
                f'[{status_map[self.tracking_mode]}] '
                f'pos({self.x_pos:.2f}, {self.y_pos:.2f}, {self.z_pos:.2f}) | '
                f'sp({self.sp_x:.2f}, {self.sp_y:.2f}, {self.sp_z:.2f})'
            )

            # Trigger landing when centred at correct altitude
            if (self._marker_visible
                    and self._last_tvec is not None
                    and abs(self._last_tvec[0]) < self.LANDING_DEADBAND
                    and abs(self._last_tvec[1]) < self.LANDING_DEADBAND
                    and abs(self._last_tvec[2] - self.LANDING_ALTITUDE)
                        < self.LANDING_DEADBAND):
                self.get_logger().info('Centred — switching to LAND mode.')
                self.stage = 6

        # Stage 6: LAND
        elif self.stage == 6:
            if self.state.mode != 'LAND':
                req = SetMode.Request()
                req.custom_mode = 'LAND'
                self.mode_client.call_async(req)
                self.get_logger().info('LAND mode commanded.')
            else:
                self.get_logger().info('LAND mode active.')
                self.timer.cancel()
                self.sp_timer.cancel()


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