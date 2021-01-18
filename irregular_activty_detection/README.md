## 베이스라인 폴더 구조
폴더 하위에 다음과 같은 구조로 데이터와 코드가 위치해야 함.

    ```
    .
    └── baseline : 베이스라인 모델이 저장되는 폴더  
    |
    └── dataset : 데이터셋이 들어갈 폴더
    |    └── train
    |    └── test
    |
    └── models : 베이스라인 모델의 백본 코드 폴더
        
    ```

## 데이터셋 폴더 구조

* train
    * 최상위 폴더내 5종의 이상행동 폴더가 존재하며 이상행동당 80개의 비디오클립으로 구성.
      해당 클립은 이상행동 영상에서 2fps로 추출된 이미지가 저장되어 있음.
      
    ```
    dataset
    └── train
        └── abandonment
        │ └ {clip_m}
        │   └── {img_n}.jpg
        │   └── ...
        │ └── ...
        ├── escalator_fall
        │ └ {clip_m}
        │   └── {img_n}.jpg
        │   └── ...
        │ └── ...
        ├── fainting
        │ └ {clip_m}
        │   └── {img_n}.jpg
        │   └── ...
        │ └── ...
        ├── surrounding_fall
        │ └ {clip_m}
        │   └── {img_n}.jpg
        │   └── ...
        │ └── ...
        ├── theft
        │ └ {clip_m}
        │   └── {img_n}.jpg
        │   └── ...
        └─└── ...  
        
    ```
  
* test
    * 최상위 폴더내 100개의 test 클립폴더로 구성되며 각 폴더내 해당 이상행동의 이미지가 존재
    ```
    dataset
    └── test
        │ └ {clip_m}
        │   └── {img_n}.jpg
        │   └── ...
        │ └ {clip_m}
        │   └── {img_n}.jpg
        │   └── ...
        │ └ {clip_m}
        │   └── {img_n}.jpg
        └───└── ...
    ```
  
## 베이스라인 코드 실행

실행 결과물로서 제출파일 샘플인 submit.json 파일 저장 (submit_sample.json은 값을 n으로 대체한 양식 파일)

* 실행방법
```
python baseline.py
```
