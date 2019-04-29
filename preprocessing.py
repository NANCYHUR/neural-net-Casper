"""
data pre-processing
"""
import pandas as pd
import numpy as np

# output_encoding matrix
# type              unit1       unit2       unit3       unit4
# scrub             0.1838      0.3174      0.3798      0.4
# dry scler         0.8162      0.3174      0.3798      0.4
# wet-dry scler     0.5         0.8651      0.3798      0.4
# wet scler         0.5         0.5         0.8872      0.4
# rain forest       0.5         0.5         0.5         0.9
output_encoding = [[0.1838, 0.3174, 0.3798, 0.4],
                  [0.8162, 0.3174, 0.3798, 0.4],
                  [0.5, 0.8651, 0.3798, 0.4],
                  [0.5, 0.5, 0.8872, 0.4],
                  [0.5, 0.5, 0.5, 0.9]]
output_types = ['SC', 'DS', 'WD', 'WS', 'RF']

def pre_process():
    # load all data
    data = pd.read_csv('GIS/gis-data.txt', sep=' ')
    # drop first column as it is identifier
    data.drop(data.columns[0], axis=1, inplace=True)
    # drop sin and cos of aspect, as they are duplicate with aspect
    data.drop('SA', axis=1, inplace=True)
    data.drop('CA', axis=1, inplace=True)

    # try shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # normalize input data: altitude, topographic position, slope degree,
    #                       rainfall, temperature, landsat band tm1-7
    for column in ['AL', 'TP', 'SL', 'RA', 'TE', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']:
        data[column] = data.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # encode aspect by 4 units (Bustos and Gedeon, 1995)
    # AS    Direction   A1      A2      A3      A4
    # 0     Flat        0       0       0       0
    # 10    N           1       0.5     0       0.5
    # 20    NE          1       1       0       0
    # 30    E           0.5     1       0.5     0
    # 40    SE          0       1       1       0
    # 50    S           0       0.5     1       0.5
    # 60    SW          0       0       1       1
    # 70    W           0.5     0       0.5     1
    # 80    NW          1       0       0       1
    a1_dict = {0: 0, 10: 1, 20: 1, 30: 0.5, 40: 0, 50: 0, 60: 0, 70: 0.5, 80: 1}
    a2_dict = {0: 0, 10: 0.5, 20: 1, 30: 1, 40: 1, 50: 0.5, 60: 0, 70: 0, 80: 0}
    a3_dict = {0: 0, 10: 1, 20: 0, 30: 0.5, 40: 1, 50: 1, 60: 1, 70: 0.5, 80: 0}
    a4_dict = {0: 0, 10: 0.5, 20: 0, 30: 0, 40: 0, 50: 0.5, 60: 1, 70: 1, 80: 1}
    data['A1'] = data['AS'].map(lambda k: a1_dict[k])
    data['A2'] = data['AS'].map(lambda k: a2_dict[k])
    data['A3'] = data['AS'].map(lambda k: a3_dict[k])
    data['A4'] = data['AS'].map(lambda k: a4_dict[k])
    # drop original 'AS' column as it's useless now
    data.drop('AS', axis=1, inplace=True)

    # encode geology descriptor by 4 units (Bustos and Gedeon, 1995)
    # GE    G1      G2      G3      G4
    # 10    0.9
    # 20    0.9
    # 30    0.9
    # 40    0.9
    # 50            0.9
    # 60    0.9
    # 70                    0.9
    # 80    0.9
    # 90                            0.9
    # (all blanks are 0.1)
    ge_encoding = {'G1': {10, 20, 30, 40, 60, 80}, 'G2': {50}, 'G3': {70}, 'G4': {90}}
    data['G1'] = data['GE'].map(lambda ge: 0.9 if ge in ge_encoding['G1'] else 0.1)
    data['G2'] = data['GE'].map(lambda ge: 0.9 if ge in ge_encoding['G2'] else 0.1)
    data['G3'] = data['GE'].map(lambda ge: 0.9 if ge in ge_encoding['G3'] else 0.1)
    data['G4'] = data['GE'].map(lambda ge: 0.9 if ge in ge_encoding['G4'] else 0.1)
    # drop original 'GE' column
    data.drop('GE', axis=1, inplace=True)

    # pre-process output, encode using equilateral coding (Bustos and Gedeon, 1995)
    data['type'] = np.nan
    for i in range(data.shape[0]):
        for j in range(len(output_types)):
            if data[output_types[j]][i] == 90:
                data.loc[i, 'type'] = j
                break
    # drop the 5 original output columns
    data.drop('SC', axis=1, inplace=True)
    data.drop('DS', axis=1, inplace=True)
    data.drop('WD', axis=1, inplace=True)
    data.drop('WS', axis=1, inplace=True)
    data.drop('RF', axis=1, inplace=True)
    # encode using equilateral coding (the above table)
    data['unit1'] = data['type'].map(lambda j: output_encoding[int(j)][0])
    data['unit2'] = data['type'].map(lambda j: output_encoding[int(j)][1])
    data['unit3'] = data['type'].map(lambda j: output_encoding[int(j)][2])
    data['unit4'] = data['type'].map(lambda j: output_encoding[int(j)][3])
    # drop 'type' column
    data.drop('type', axis=1, inplace=True)

    return data


# according to equilateral encoding of the output,
# retrieve the category by calculating Euclidean distance
def interpret_output(output_vertex):
    closest_type = None
    closest_distance = np.inf
    for i in range(len(output_encoding)):
        type = output_encoding[i]
        dist = np.linalg.norm(np.array(output_vertex) - np.array(type))
        if closest_distance > dist:
            closest_distance = dist
            closest_type = output_types[i]
    return closest_type
