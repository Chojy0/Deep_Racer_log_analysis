# dr_log_analysis
AWS DeepRacer 출전 당시 사용한 log 분석이 가능한 코드 모음

## ✅ 프로젝트 한눈에 보기 
이 레포지토리는 AWS DeepRacer 주행 로그를 분석하여 **에피소드 성능, 보상 분포, 속도/조향 특성** 등을 빠르게 파악할 수 있도록 구성되어 있습니다.

### 빠른 시작
```bash
python deepracer_report.py \
  --log-folder center \
  --track-file reInvent2019_track_ccw.npy \
  --output-dir report_output
```

### 입력 데이터 위치
- 기본 경로: `log_analysis/<폴더명>/training-simtrace`
  - 예: `log_analysis/center/training-simtrace`
- 트랙 파일 경로: `tracks/`

### 출력 결과
`report_output/` 아래에 다음 파일이 생성됩니다.
- `overall_summary.json`: 전체 요약 지표
- `episode_summary.csv`: 에피소드별 요약 지표
- `report.md`: 간단 리포트
- `plots/*.png`: 그래프 이미지들
