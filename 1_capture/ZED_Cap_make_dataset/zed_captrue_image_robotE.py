import sys
import pyzed.sl as sl
import cv2
import threading
import time
import os
import mecademicpy.robot as mdr

# 데이터 저장 디렉토리 설정
OUTPUT_DIR = "./vla_dataset"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_FILE = os.path.join(OUTPUT_DIR, "robot_data.txt")  # 로봇 데이터만 저장

# 파일 쓰기 잠금 객체
file_lock = threading.Lock()

# 카메라 클래스 정의
class ZedCamera:
    def __init__(self, serial_number, output_dir):
        self.serial_number = serial_number
        self.output_dir = os.path.join(OUTPUT_DIR, output_dir)
        self.zed = sl.Camera()
        self.running = False
        self.ready = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.init_camera()

    def init_camera(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 30
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.QUALITY

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open camera {self.serial_number}: {err}")
            sys.exit(-1)
        else:
            print(f"Camera {self.serial_number} opened successfully")

            runtime_params = sl.RuntimeParameters()
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.ready = True
            else:
                print(f"Camera {self.serial_number} failed to grab initial frame")

    def capture_and_save(self, runtime_params, start_event, duration=30):
        self.running = True
        left_image = sl.Mat()
        right_image = sl.Mat()

        # 시작 이벤트 대기
        start_event.wait()
        start_time = time.time()
        print(f"Camera {self.serial_number} started capturing")

        while self.running and (time.time() - start_time) < duration:
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                timestamp = time.time()
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                left_data = left_image.get_data()
                right_data = right_image.get_data()

                if left_data is None or right_data is None:
                    print(f"Camera {self.serial_number} - Failed to capture data at {timestamp:.3f}")
                    continue

                # 타임스탬프를 소수점 3자리로 고정
                timestamp_str = f"{timestamp:.3f}"
                left_image_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_left_{timestamp_str}.jpg")
                right_image_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_right_{timestamp_str}.jpg")

                if not (cv2.imwrite(left_image_path, left_data[:, :, :3]) and cv2.imwrite(right_image_path, right_data[:, :, :3])):
                    print(f"Camera {self.serial_number} - Failed to save images at {timestamp:.3f}")

            time.sleep(0.1)  # 0.1초 간격으로 데이터 수집

        self.zed.close()
        print(f"Camera {self.serial_number} stopped")

    def stop(self):
        self.running = False

# 로봇 데이터 수집 스레드
def collect_robot_data(robot, data_file, start_event, duration=30):
    start_event.wait()
    start_time = time.time()
    print("Robot data collection started")

    while time.time() - start_time < duration:
        timestamp = time.time()
        joint_angles = robot.GetJoints()
        cartesian_pose = robot.GetPose()

        joint_angles_str = ",".join([f"{angle:.2f}" for angle in joint_angles])
        cartesian_pose_str = ",".join([f"{coord:.2f}" for coord in cartesian_pose])

        # 타임스탬프를 소수점 3자리로 고정
        timestamp_str = f"{timestamp:.3f}"
        with file_lock:
            data_file.write(f"{timestamp_str},{joint_angles_str},{cartesian_pose_str}\n")
            data_file.flush()

        time.sleep(0.1)  # 0.1초 간격으로 데이터 수집

    print("Robot data collection stopped")

# 메인 실행 함수
def run_robot_and_cameras(robot_ip="192.168.0.100", duration=30):
    # 로봇 객체 생성
    robot = mdr.Robot()

    # 카메라 설정
    cameras = [
        ZedCamera(serial_number=41182735, output_dir="front"),
        ZedCamera(serial_number=49429257, output_dir="right"),
        ZedCamera(serial_number=44377151, output_dir="left"),
        ZedCamera(serial_number=49045152, output_dir="top"),
    ]

    # 모든 카메라가 준비될 때까지 대기
    print("Waiting for all cameras to be ready...")
    while not all(camera.ready for camera in cameras):
        time.sleep(1)
    print("All cameras are ready")

    # 데이터 파일 열기
    with open(DATA_FILE, "w") as data_file:
        data_file.write("timestamp,joint_angles,cartesian_pose\n")

        # 시작 이벤트 생성
        start_event = threading.Event()

        # 카메라 스레드 시작
        runtime_params = sl.RuntimeParameters()
        threads = []
        for camera in cameras:
            thread = threading.Thread(target=camera.capture_and_save, args=(runtime_params, start_event, duration))
            threads.append(thread)
            thread.start()

        # 로봇 데이터 수집 스레드 시작
        robot_thread = threading.Thread(target=collect_robot_data, args=(robot, data_file, start_event, duration))
        robot_thread.start()

        try:
            # 로봇 연결 및 초기화
            print(f"Connecting to robot at {robot_ip}...")
            robot.Connect(address=robot_ip, enable_synchronous_mode=False)
            robot.ActivateAndHome()
            robot.WaitHomed()
            print("Robot activated and homed successfully")
            robot.SetJointVel(1)
            print("Joint velocity set to 3 deg/s")

            # 로봇과 카메라 동시 시작 준비
            print("Starting robot sequence and camera capture...")
            start_event.set()  # 모든 스레드에 시작 신호 전송
            start_time = time.time()

            # 로봇 동작 (별도 루프)
            poses = [
                    (185, 98, 195, -156, -2, 123),
                    (185, 110, 150, -156, -2, 123),
                    (185, 98, 195, -156, -2, 123)
            ]
            pose_index = 0

            while time.time() - start_time < duration:
                # 로봇 동작
                x, y, z, rx, ry, rz = poses[pose_index % len(poses)]
                robot.MovePose(x, y, z, rx, ry, rz)
                robot.WaitIdle()
                pose_index += 1

                time.sleep(0.01)  # 로봇 동작 루프도 0.1초 간격 유지

        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            for camera in cameras:
                camera.stop()
            for thread in threads:
                thread.join()
            robot_thread.join()
            robot.DeactivateRobot()
            robot.WaitDeactivated()
            robot.Disconnect()
            print("Robot and cameras disconnected")

if __name__ == "__main__":
    try:
        run_robot_and_cameras(robot_ip="192.168.0.100", duration=30)
    except KeyboardInterrupt:
        print("Interrupted by user")