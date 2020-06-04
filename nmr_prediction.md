# NMR prediction task


- Given input : SMILES input as string
- Training : SMILES to NMR peak

- output : transformer output
- output은 gather()된 뒤 RMSE error를 출력하도록 함
- test set도 설정

- NMR dataset : given as molecule(RDkit) and index.
  - Check가 필요함.

