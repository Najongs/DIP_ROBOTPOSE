import pyzed.sl as sl
import cv2
import threading
import time
import os
import json
import sys

# .so 파일 경로 설정
sys.path.append("/home/intertek/DGIST/fairino")
from Robot import RPC

# 로봇 객체 생성
robot = RPC('192.168.58.2')

# 저장 폴더
OUTPUT_DIR = "./Fr5_intertek"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./Fr5_intertek/joint", exist_ok=True)

# ✅ 로봇 조인트 정보 수집 함수
def GetActualJointPosDegree(flag=1):
    try:
        _error = robot.GetActualJointPosDegree(int(flag))
        if _error[0] == 0:
            return _error[1]
        else:
            return None
    except Exception as e:
        print(f"Exception in GetActualJointPosDegree: {e}")
        return None


class ZedCamera(threading.Thread):
    def __init__(self, serial_number, output_subdir, start_event, duration=30):
        super().__init__()
        self.serial_number = serial_number
        self.output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.start_event = start_event
        self.duration = duration
        self.ready = False

    def init_camera(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.ready = True
            print(f"Camera {self.serial_number} initialized")
        else:
            print(f"Failed to open camera {self.serial_number}")
            sys.exit(1)

    def run(self):
        if not self.ready:
            print(f"Camera {self.serial_number} not ready. Skipping capture.")
            return

        left_image = sl.Mat()
        right_image = sl.Mat()

        self.start_event.wait()
        print(f"Camera {self.serial_number} started capturing")
        start_time = time.time()

        while time.time() - start_time < self.duration:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                timestamp = time.time()
                timestamp_str = f"{timestamp:.3f}"

                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                left_data = left_image.get_data()
                right_data = right_image.get_data()

                left_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_left_{timestamp_str}.jpg")
                right_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_right_{timestamp_str}.jpg")

                success_left = cv2.imwrite(left_path, left_data[:, :, :3])
                success_right = cv2.imwrite(right_path, right_data[:, :, :3])

                if not (success_left and success_right):
                    print(f"Camera {self.serial_number} - Failed to save images at {timestamp:.3f}")

                # joint json 파일은 통합 폴더(OUTPUT_DIR)에 저장
                joint = GetActualJointPosDegree()
                if joint:
                    joint_path = os.path.join("./Fr5_intertek/joint", f"joint_{self.serial_number}_{timestamp_str}.json")
                    with open(joint_path, 'w') as f:
                        json.dump(joint, f, indent=4)
                else:
                    print(f"Camera {self.serial_number} - Failed to get joint at {timestamp:.3f}")

            time.sleep(0.1)

        self.zed.close()
        print(f"Camera {self.serial_number} stopped")


def main():
    start_event = threading.Event()
    duration = 30

    cameras = [
        ZedCamera(serial_number=34850673, output_subdir="right", start_event=start_event, duration=duration),
        ZedCamera(serial_number=38007749, output_subdir="left", start_event=start_event, duration=duration),
        ZedCamera(serial_number=30779426, output_subdir="top", start_event=start_event, duration=duration),
    ]

    for cam in cameras:
        cam.init_camera()

    while not all(cam.ready for cam in cameras):
        print("Waiting for all cameras to be ready...")
        time.sleep(1)

    for cam in cameras:
        cam.start()

    print("Starting data capture in 3 seconds...")
    time.sleep(3)
    start_event.set()

    for cam in cameras:
        cam.join()

    print("Data collection finished.")


if __name__ == "__main__":
    main()
