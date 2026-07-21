# CLAUDE.md — DIP 워크스페이스 규칙

로봇 포즈 추정 연구 모노레포. 전체 구조는 [README.md](README.md), 카테고리별 실험 방법은 [docs/](docs/README.md) 참조.

## 구조 요약

기능 흐름 순 카테고리: `1_capture/`(데이터 수집·캘리브) → `2_robot/`(FK·DH 유틸) → `3_pose_models/`(포즈 학습) → `4_perception/`(세그·뎁스·통합) → `5_apps/`(응용). 데이터는 `datasets/`, 공용 노트북은 `notebooks/`.

## git 규칙

- **이 repo(모노레포)가 유일한 git. push 대상은 `origin = Najongs/DIP_ROBOTPOSE` 뿐.**
- 과거 프로젝트별 GitHub repo(`Najongs/DINObotPose3` 등)는 읽기 전용 아카이브 — 절대 push하지 말 것. 단, GPU 서버 작업을 가져올 땐 그 repo를 fetch 경유해 수동 반영.
- 코드·문서·설정만 추적. 미디어(*.png/jpg/json), 가중치(*.pt/pth), zip, 학습 산출물(wandb/, outputs*/, results*/)은 .gitignore — 예외 추가 전 용량 확인.
- `4_perception/DINOv3_fine_tunning/{dinov3,Depth-Anything-3}/`는 외부 공식 repo 클론(자체 .git 보유) — 건드리지 말 것.
- `.gitignore`가 `*.json`/`*.png`/`*.jpg`/`*.pdf`를 통째로 무시 — 설정·figure를 커밋하려면 `git add -f` 필요. (현재 tracked `.json`은 0개, `docs/dinobotpose3/figures/`의 PDF 12개는 전부 force-add된 것)
- 루트의 `DREAM/`, `RoboPEPP/`, `robopose/`는 미추적 경쟁모델 클론이며 **각자 `.git` 보유** — 루트에서 `git add -A` 하면 bogus gitlink가 생김. (`_assets_src/`만 gitignore돼 있고 이 셋은 아님)
- 커밋 제목은 타입 접두사(`docs:`/`feat:`/`fix:`/`paper:`/`chore:`), 본문은 영어. 문서·주석은 한국어.

## 경로 규칙

- 전 코드가 새 구조 절대경로(`/home/najo/NAS/DIP/<카테고리>/<프로젝트>/...`)로 통일됨 (2026-07-03 일괄 치환).
- `datasets/`로 향하는 프로젝트 심볼릭링크가 **8개** 있음 (ICRA `dataset`, DINObotPose3 `Dataset/*`, 1_capture 캡처들) — **전부 제거 금지.** ICRA 코드는 `__file__` 기준으로 `<프로젝트>/dataset`을 계산함. 이름 비대칭 주의: `DINObotPose3/Dataset/Converted_dataset/DREAM_real` → `.../Converted_dataset/DREAM_to_DREAM`.
- 루트의 `yolov8n-seg.pt`, `yolo_train_robot_box.yaml`은 코드가 루트 기준으로 참조 — 이동 금지.
- 새 코드는 절대경로 하드코딩 대신 CLI 인자/설정 파일 사용.

## 데이터 보호

- `datasets/`(~44G)와 `1_capture/DGIST_IROM_Data_collection/`은 **NAS가 유일본 — 삭제·이동 금지.** 지도는 [docs/datasets.md](docs/datasets.md).
- `4_perception/Fr5_robot_SegFormer/best_segformer_robot_arm.pth`는 재생성 데이터가 로컬에 없음 — 삭제 금지 (collision_risk_pipeline이 사용).
- 학습 산출물(wandb/, outputs*/, eval_outputs*/)은 재생성 가능 — 용량 정리 시 삭제 대상 1순위.

## 실행 환경 (이 머신)

- **GPU는 UUID로 지정.** `nvidia-smi` 인덱스와 `CUDA_VISIBLE_DEVICES` 정수 인덱스가 어긋남 (실측: smi idx0=3090인데 `CUDA_VISIBLE_DEVICES=0`은 A6000). 항상 `CUDA_VISIBLE_DEVICES=GPU-<uuid>`.
- GPU 5장이 상시 100% util이지만 메모리는 여유 있음 — **util이 아니라 free memory 기준으로 고를 것.**
- conda env는 `/home/najo/.conda/envs/`에 있음 (`/opt/anaconda3/envs`는 비어 있음). DINObotPose3 = `dino`(py3.10, torch 2.10+cu128). 스크립트에선 `conda activate` 대신 절대경로 인터프리터 권장: `/home/najo/.conda/envs/dino/bin/python`.
- `dinov3` env는 **존재하지 않음** — `4_perception/DINOv3_fine_tunning/`의 셸 스크립트 6개가 이걸 참조해 현재 실행 불가.

## GPU 서버 워크플로우

- DINObotPose3 실험은 GPU 서버(`/data/public/NAS/...`)에서 진행됨. 그쪽 체크포인트·최신 커밋이 로컬에 없을 수 있음.
- 동기화 절차: GPU 서버 → GitHub `Najongs/DINObotPose3` push → 로컬에서 fetch 후 모노레포 경로 규칙에 맞게 수동 반영 (예: bdd0fc1 커밋 방식).
- DINObotPose3 스크립트에 GPU 서버 경로(`/data/public/NAS/...`)가 남아 있는 것은 의도적 — 로컬 경로로 바꾸지 말 것.

## 실험 컨벤션

- DINObotPose3 실험은 `EXPERIMENTS.md`(일지)와 `SUMMARY.md`(확정 결론·REFUTED 목록)에 기록 — **새 실험 전 SUMMARY의 REFUTED 목록 확인** (백본 SSL 적응, co-finetune, union-bbox 등은 이미 반증됨).
- **배포 설정의 유일 authority는 `docs/dinobotpose3/FINAL_MODEL.md`.** `Eval/verify_sota.sh`와 `docs/.../training/training.md`의 체크포인트 표는 구버전(0.799)임.
- first-party 코드에 테스트·린트 설정 없음. DINObotPose3는 `/run-dinobotpose3` 스킬의 `driver.py fwd`(9초) / `smoke`(80초)가 사실상 유일한 회귀 게이트.
- 학습은 wandb 로깅 사용이 관례 (프로젝트명은 스크립트 상단 참조).
- 문서 갱신: 프로젝트 구조나 실행 방법이 바뀌면 `docs/<카테고리>.md`도 함께 갱신.

## 논문 (DINObotPose3)

- 본문 `docs/dinobotpose3/PAPER_OVERLEAF.tex`. 컴파일은 **Overleaf에서** — repo에 pdflatex/latexmk 호출 없음.
- figure 재생성(env `dino`): `python docs/dinobotpose3/figures/make_figs.py` (+ `make_fig_pipeline.py`, `make_figs_multirobot.py`)

## 활성/유휴 상태 (2026-07 기준)

- 활성: DINObotPose3(논문 집필 + 실험 — **로컬에서도 전체 평가 실행 가능**), `5_apps/collision_risk_pipeline`, `4_perception/Fr5_robot_SegFormer`
- 유휴: 나머지 (ICRA, Meca500, 캡처 프로젝트들 — 코드는 재사용 가치 있음, docs 참조)
