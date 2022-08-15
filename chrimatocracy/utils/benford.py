import numpy as np
import pandas as pd


def find_leading_number(line):
    numbers = "123456789"
    line = str(line)
    index = len(line)
    for i in range(0, index):
        if line[i] in numbers:
            return int(line[i])
    return 0


def read_numbers(dataFile):
    listOfoccurrances = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for lines in dataFile:
        lead = find_leading_number(lines)
        if lead != 0:
            listOfoccurrances[lead - 1] += 1
    return listOfoccurrances


def find_probabilities(listOfoccurrances):
    total = 0
    probabilities = []
    for number in listOfoccurrances:
        total += number
    for number in listOfoccurrances:
        value = float(number) / total
        probabilities.append(round(value, 3))
    return probabilities


def find_Cho(listOfoccurrances):
    choList = []
    probs = find_probabilities(listOfoccurrances)
    N = listOfoccurrances.sum()
    for i in range(0, 9):
        benford = np.log10(1 + (1.0 / (i + 1)))
        d = np.square(benford - probs[i])
        choList.append(d)
    return np.sqrt(N * sum(choList))


def find_x2(listOfoccurrances):
    x2List = []
    probs = find_probabilities(listOfoccurrances)
    N = listOfoccurrances.sum()
    for i in range(0, 9):
        benford = np.log10(1 + (1.0 / (i + 1)))
        x2 = np.square(benford - probs[i]) / benford
        x2List.append(x2)
    return N * sum(x2List), x2List


def benford_digits_table(donations, by):
    lista = donations.groupby(by).size()

    df = pd.DataFrame(index=lista.keys(), columns=("prob", "d", "N"))
    for ele in lista.keys():
        digits_all = read_numbers(donations[(donations[by] == ele)]["num_donation_ammount"])
        digits_all = pd.Series(digits_all)
        probs = find_probabilities(digits_all)
        indx = np.arange(1, len(probs) + 1)
        benford = [np.log10(1 + (1.0 / d)) for d in indx]
        N = digits_all.sum()
        d = find_Cho(digits_all)
        df.loc[ele] = [probs, d, N]

    df[["1", "2", "3", "4", "5", "6", "7", "8", "9"]] = df["prob"].apply(pd.Series)
    df_repass_legal = (
        df.drop(["prob"], axis=1).dropna().sort_index(axis=1).sort_values(by=["d"], ascending=False).reset_index()
    )

    return df_repass_legal
