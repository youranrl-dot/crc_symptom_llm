# GitHub 업로드 절차

현재 레포(`youranrl-dot/crc_symptom_llm`)에는 6월 12일자 01~07 파이프라인이 올라가 있고,
17분 전에 올리신 README.md는 **레포에 없는 파일들을 참조하고 있어 지금 깨진 상태**입니다.
아래 순서대로 하면 정리됩니다.

---

## 1단계 — 옛 스크립트 4개 삭제

레포 웹 화면에서 각 파일을 열고 → 오른쪽 위 **휴지통 아이콘** → 맨 아래 `Commit changes`.

- `03_ner_based.py`
- `04_evaluate.py`
- `06_network_analysis.py`
- `07_predictive_validity.py`

> 지워도 안전합니다. Git이 이력을 보관하므로 `Commits` 탭에서 언제든 옛 버전을 볼 수 있습니다.
> 이 넷은 200노트 gold가 아닌 1,000노트 파일로, 46개가 아닌 41개 증상으로 채점했고,
> NER이라 이름 붙었지만 실제로는 키워드 매처였으며, 이번에 고친 outcome 정의 이전 버전입니다.
> 남겨두면 누군가 그대로 돌릴 위험이 있어 삭제를 권합니다.

## 2단계 — 그대로 두는 파일 (4개)

`01_extract_crc_notes.py` · `02_rule_based.py` · `05_llm_extraction.py` ·
`symptom_dict.py` · `requirements.txt`

## 3단계 — 새로 올릴 파일

`Add file` → `Upload files`로 아래를 올립니다. **`analysis` 폴더는 폴더째 드래그**하면
경로가 유지됩니다.

```
README.md                      ← 덮어쓰기 (지금 깨진 것 교체)
ANALYSIS.md
.gitignore                     ← 새로 포함됨 (4단계 불필요)
symptoms.json
requirements-analysis.txt
compare_with_gold.py
run_hybrid.py
negation_filter_config.py
run_analysis.py
validate.py
analysis/                      ← 폴더째 (9개 파일)
```

레포 밖에서 따로 챙겨 올려야 하는 것 2개:

```
system_prompt.txt              ← 프로젝트에 있는 46라벨 + 코딩규칙 20개 프롬프트
run_ner_scispacy.py            ← files/ 폴더에 있는 scispaCy NER 스크립트
```

## 4단계 — `.gitignore`

패키지에 `.gitignore`가 포함되어 있으니 3단계에서 같이 올리시면 됩니다.
**MIMIC-IV DUA상 노트 텍스트가 커밋되면 안 되므로 이 파일은 필수입니다.**
GitHub 웹 업로드에서 점으로 시작하는 파일이 보이지 않으면, `Add file` →
`Create new file` → 파일명 `.gitignore` 로 직접 만들고 패키지의 내용을 붙여넣으세요.

## 5단계 — 확인

삭제·업로드가 끝나면 레포 첫 화면이 이렇게 보이면 됩니다.

```
.gitignore                 ANALYSIS.md              README.md
analysis/                  01_extract_crc_notes.py  02_rule_based.py
05_llm_extraction.py       compare_with_gold.py
negation_filter_config.py  run_analysis.py          run_hybrid.py
run_ner_scispacy.py        symptom_dict.py          symptoms.json
system_prompt.txt          validate.py
requirements.txt           requirements-analysis.txt
```

---

## 올리기 전에 알아두실 것

새 README의 **Known issues** 절에 여섯 가지를 공개 기록해뒀습니다. 숨기는 것보다
명시하는 편이 재현성 심사에서 유리합니다.

1. 삭제한 옛 스크립트 4개는 현재 원고를 재현하지 못함 (`04_evaluate.py`는 200노트 gold가
   아니라 1,000노트 파일로, 46개가 아니라 41개 증상으로 채점) — git 이력에는 남아 있음
2. `01`·`02`·`03`·`05`에 개발 당시 **절대경로가 하드코딩**되어 있어 다른 환경에서 실행 불가
3. `05_llm_extraction.py` 독스트링이 **Claude Haiku / GPT-4o mini / Gemini 1.5 Flash**로
   되어 있는데, 원고는 Claude Haiku(`claude-haiku-4-5-20251001`)와
   Gemini **3.5** Flash를 쓰고 GPT는 보고하지 않음
4. Gemini 추론 결과가 두 벌이고 200노트에서 라벨 93개(1.0%)가 불일치
5. Claude 출력의 18개 노트가 `error=missing_cc_hpi`로 전부 null
6. NER 행(0.3842)을 재현하는 파일 없음

2번과 3번은 **제가 검증할 수 없는 부분**이라 손대지 않았습니다. 절대경로만 인자로
바꾸시면 되고, 3번은 실제 쓰신 모델 버전으로 독스트링만 고치시면 됩니다.

`analysis/`, `compare_with_gold.py`, `run_hybrid.py`, `validate.py`는 실제 데이터로 돌려서
**58개 검증 항목 전부 통과**를 확인했습니다. Table 2·3의 6개 행이 전부 이 코드로 재현됩니다.
