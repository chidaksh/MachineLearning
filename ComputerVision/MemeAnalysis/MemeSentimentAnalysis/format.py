import pandas as pd

df = pd.read_csv("memotion_val.csv")
with open("answer.txt", "w") as f:
    for n, row in df.iterrows():
        a, b, c = 0, [], []
        if row["overall_sentiment"] == "neutral":
            a = "1"
        elif row["overall_sentiment"] == "negative" or row["overall_sentiment"] == "very_negative":
            a = "0"
        else:
            a = "2"

        if row["humour"] == "not_funny":
            b.append("0")
        else:
            b.append("1")
        if row["sarcastic"] == "not_sarcastic":
            b.append("0")
        else:
            b.append("1")
        if row["offensive"] == "not_offensive":
            b.append("0")
        else:
            b.append("1")
        if row["motivational"] == "not_motivational":
            b.append("0")
        else:
            b.append("1")
        b = "".join(b)

        if row["humour"] == "not_funny":
            c.append("0")
        elif row["humour"] == "funny":
            c.append("1")
        elif row["humour"] == "very_funny":
            c.append("2")
        else:
            c.append("3")
        if row["sarcastic"] == "not_sarcastic":
            c.append("0")
        elif row["sarcastic"] == "little_sarcastic":
            c.append("1")
        elif row["sarcastic"] == "very_sarcastic":
            c.append("2")
        else:
            c.append("3")
        if row["offensive"] == "not_offensive":
            c.append("0")
        elif row["offensive"] == "slight":
            c.append("1")
        elif row["offensive"] == "very_offensive":
            c.append("2")
        else:
            c.append("3")
        if row["motivational"] == "not_motivational":
            c.append("0")
        else:
            c.append("1")
        c = "".join(c)
        f.write(a + "_" + b + "_" + c + "\n")