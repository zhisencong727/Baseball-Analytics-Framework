def expectation(winProb,tieProb,odds):
    if odds < 0:
        payout = winProb * ((odds-100)/odds) + tieProb
    else:
        payout = winProb * ((odds+100)/100) + tieProb
    return payout

def main():
    while True:
        winProb = input("What is model's winProb?")
        tieProb = input("What is model's tieProb")
        odds = input("What is the sportsbook's odds?")
        print("Expectation is: ",round(expectation(float(winProb),float(tieProb),int(odds)),3))
        cont = input("Continue?")
        if cont == "0":
            break
main()