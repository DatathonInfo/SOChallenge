# Baseline code
주제2. 수식/도형/낙서/기호/OCR 데이터 중 인쇄체 인식 모델 개발

## 실행 방법

```bash
# 명칭이 'CUBOX_OCR'인 데이터셋을 사용해 세션 실행하기
$ nsml run -d CUBOX_OCR
# 메인 파일명이 'main.py'가 아닌 경우('-e' 옵션으로 entry point 지정)
# 예: nsml run -d CUBOX_OCR -e main.py
$ nsml run -d CUBOX_OCR -e [파일명]

# 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml logs -f [세션명]

# 세션 종료 후 모델 목록 및 제출하고자 하는 모델의 checkpoint 번호 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml model ls [세션명]

# 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
$ nsml submit -t [세션명] [모델_checkpoint_번호]

# 모델 제출하기
# 제출 후 리더보드에서 점수 확인 가능
nsml submit [세션명] [모델_checkpoint_번호]
```
