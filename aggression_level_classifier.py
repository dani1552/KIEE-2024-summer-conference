import pandas as pd

df = pd.read_csv('assets/metadata_labels_v3.csv')

# 각 열의 값들을 리스트로 추출
BPAQ_Hostility_Label = df['BPAQ_Hostility_Label'].tolist()
BPAQ_VerbalAggression_Label = df['BPAQ_VerbalAggression_Label'].tolist()
BPAQ_Anger_Label = df['BPAQ_Anger_Label'].tolist()
BPAQ_PhysicalAggression_Label = df['BPAQ_PhysicalAggression_Label'].tolist()


# 1) csv 파일 따로 저장
pd.DataFrame({'BPAQ_Hostility_Label': BPAQ_Hostility_Label}).to_csv('BPAQ_Hostility_Label.csv', index=False)
pd.DataFrame({'BPAQ_VerbalAggression_Label': BPAQ_VerbalAggression_Label}).to_csv('BPAQ_VerbalAggression_Label.csv', index=False)
pd.DataFrame({'BPAQ_Anger_Label': BPAQ_Anger_Label}).to_csv('BPAQ_Anger_Label.csv', index=False)
pd.DataFrame({'BPAQ_PhysicalAggression_Label': BPAQ_PhysicalAggression_Label}).to_csv('BPAQ_PhysicalAggression_Label.csv', index=False)


# # 2) csv 파일 하나로 저장
# combined_df = pd.DataFrame({
#     'BPAQ_Hostility_Label': BPAQ_Hostility_Label,
#     'BPAQ_VerbalAggression_Label': BPAQ_VerbalAggression_Label,
#     'BPAQ_Anger_Label': BPAQ_Anger_Label,
#     'BPAQ_PhysicalAggression_Label': BPAQ_PhysicalAggression_Label
# })

# # 결합된 데이터프레임을 CSV 파일로 저장
# combined_df.to_csv('combined_BPAQ_labels.csv', index=False)
