# -*-coding:utf-8-*-
# Lab 2 - Robotic Arm Keyboard Control
#
# Controls the two-link robotic arm and gripper via keyboard input.
#
# Arrow Keys:
#   UP    -> move arm up   (y +)
#   DOWN  -> move arm down (y -)
#   RIGHT -> extend arm forward (x +)
#   LEFT  -> retract arm back   (x -)
#
# Letter Keys:
#   o -> open gripper
#   c -> close gripper
#   q -> quit
#
# Gripper power is fixed at 60.

import sys
import time
import threading
import robomaster
from robomaster import robot

# ── Configuration ────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.50.116"
ROBOT_SN   = "3JKCH8800100VW"

MOVE_STEP      = 20    # mm per key press
GRIPPER_POWER  = 60    # gripper strength (0-100)
GRIPPER_HOLD   = 1.0   # seconds to hold before pausing

# ── Platform-safe single-character keyboard read ─────────────────────────────
try:
    import tty, termios

    def _get_key():
        """Blocking single-keypress read (Unix/macOS)."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            # Arrow keys send a 3-byte escape sequence: \x1b [ A/B/C/D
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                return ch + ch2 + ch3
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

except ImportError:
    # Windows fallback using msvcrt
    import msvcrt

    def _get_key():
        """Blocking single-keypress read (Windows)."""
        ch = msvcrt.getwch()
        if ch in ('\x00', '\xe0'):          # extended / special key prefix
            ch2 = msvcrt.getwch()
            # Arrow key scan codes: 72=up, 80=down, 77=right, 75=left
            _map = {'\x48': '\x1b[A', '\x50': '\x1b[B',
                    '\x4d': '\x1b[C', '\x4b': '\x1b[D'}
            return _map.get(ch2, '')
        return ch


# ── Arrow key escape sequences ────────────────────────────────────────────────
KEY_UP    = '\x1b[A'
KEY_DOWN  = '\x1b[B'
KEY_RIGHT = '\x1b[C'
KEY_LEFT  = '\x1b[D'


# ── Gripper state helper ──────────────────────────────────────────────────────
class GripperController:
    """Wraps the gripper so open/close calls don't block the input loop."""

    def __init__(self, ep_gripper):
        self._gripper = ep_gripper
        self._lock    = threading.Lock()
        self._state   = None   # 'open' | 'closed' | None

    def open(self):
        with self._lock:
            if self._state == 'open':
                return
            self._state = 'open'
        t = threading.Thread(target=self._do_open, daemon=True)
        t.start()

    def close(self):
        with self._lock:
            if self._state == 'closed':
                return
            self._state = 'closed'
        t = threading.Thread(target=self._do_close, daemon=True)
        t.start()

    def _do_open(self):
        self._gripper.open(power=GRIPPER_POWER)
        time.sleep(GRIPPER_HOLD)
        self._gripper.pause()

    def _do_close(self):
        self._gripper.close(power=GRIPPER_POWER)
        time.sleep(GRIPPER_HOLD)
        self._gripper.pause()


# ── Main ──────────────────────────────────────────────────────────────────────
def print_help():
    print("\n" + "=" * 50)
    print("  RoboMaster Arm Keyboard Control")
    print("=" * 50)
    print("  Arrow Keys:")
    print("    ↑  UP    - Move arm UP")
    print("    ↓  DOWN  - Move arm DOWN")
    print("    →  RIGHT - Extend arm FORWARD")
    print("    ←  LEFT  - Retract arm BACK")
    print()
    print("  Gripper (power = {}):".format(GRIPPER_POWER))
    print("    o        - Open gripper")
    print("    c        - Close gripper")
    print()
    print("    q        - Quit")
    print("=" * 50 + "\n")


def main():
    # ── Connect ──────────────────────────────────────────────────────────────
    robomaster.config.ROBOT_IP_STR = ROBOT_IP
    ep_robot = robot.Robot()

    print("Connecting to RoboMaster EP...")
    try:
        ep_robot.initialize(conn_type="sta", sn=ROBOT_SN)
        print("Connected.\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    ep_arm     = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    gripper    = GripperController(ep_gripper)

    print_help()

    # ── Control loop ─────────────────────────────────────────────────────────
    try:
        while True:
            key = _get_key()

            if key == KEY_UP:
                print(f"↑  Arm UP    (y +{MOVE_STEP} mm)")
                ep_arm.move(x=0, y=MOVE_STEP).wait_for_completed()

            elif key == KEY_DOWN:
                print(f"↓  Arm DOWN  (y -{MOVE_STEP} mm)")
                ep_arm.move(x=0, y=-MOVE_STEP).wait_for_completed()

            elif key == KEY_RIGHT:
                print(f"→  Arm FWD   (x +{MOVE_STEP} mm)")
                ep_arm.move(x=MOVE_STEP, y=0).wait_for_completed()

            elif key == KEY_LEFT:
                print(f"←  Arm BACK  (x -{MOVE_STEP} mm)")
                ep_arm.move(x=-MOVE_STEP, y=0).wait_for_completed()

            elif key.lower() == 'o':
                print("  Gripper OPEN")
                gripper.open()

            elif key.lower() == 'c':
                print("  Gripper CLOSE")
                gripper.close()

            elif key in ('q', 'Q', '\x03'):   # q, Q, or Ctrl-C
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Pausing gripper and disconnecting...")
        try:
            ep_gripper.pause()
        except Exception:
            pass
        ep_robot.close()
        print("Done.")


if __name__ == '__main__':
    main()
