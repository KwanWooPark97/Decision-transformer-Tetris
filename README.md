해본 수업 프로젝트 중 가장 큰 스케일이였음.

Transformer에 교수님이 관심을 보이셔서 책을 읽으며 세미나 준비를 하며 공부를 해둠
Transformer를 강화 학습에 적용한 논문은 있나? 찾아보니 이미 Decision Transformer가 존재하는 것을 깨달음
처음 해보는 Offline learning이므로 어떤 데이터가 좋을지 고민했는데 논문으로 만든 테트리스 환경이 떠오름
학습한 모델을 이용해 약 100만개? 정도 학습 데이터 모으고 10만개 test 데이터로 성능 확인 진행
학습 데이터가 메모리 문제가 있어서 한번에 10만개씩 담겨있는 파일 10개로 나눠버림

바닐라 DT는 네트워크가 너무 커서 학습 진행이 불가능함.
찾아보니 casual DT가 존재하는 것을 발견 후 코드를 가져와 네트워크 구조를 내 환경에 맞게 수정
하지만 데스크 탑 성능이 높진 않아서 큰 네트워크를 만들 수가 없음...

일단 그냥 학습 진행

학습은 정확도 99%까지 잘 작동함
BUT test에서 정확도 0%를 보여줌 말이 안됨. 같은 환경을 사용하여 test를 진행하고 100만개로 학습을 했는데도 부족한가? 실패 원인을 찾을 수가 없음 코드 적으로 문제가 있나 계속 살펴 보고 디버깅 진행했지만
문제를 전혀 찾을 수가 없음 일단 연구 프로젝트가 우선이므로 보류