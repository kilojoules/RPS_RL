## Example Episodes

What does play look like at different strategy distributions?

### Near-Nash agent vs Near-Nash opponent

Agent policy: R=0.34  P=0.33  S=0.33
Opponent policy: R=0.33  P=0.34  S=0.33

```
Round   Agent  Opponent  Result  Payoff  Cumulative
-------------------------------------------------------
    1   Paper      Rock       W      +1          +1
    2  Scissors      Rock       L      -1          +0
    3  Scissors     Paper       W      +1          +1
    4   Paper     Paper       D      +0          +1
    5    Rock      Rock       D      +0          +1
    6    Rock     Paper       L      -1          +0
    7    Rock      Rock       D      +0          +0
    8  Scissors      Rock       L      -1          -1
    9   Paper     Paper       D      +0          -1
   10  Scissors     Paper       W      +1          +0
   11    Rock  Scissors       W      +1          +1
   12  Scissors      Rock       L      -1          +0
   13  Scissors     Paper       W      +1          +1
   14    Rock     Paper       L      -1          +0
   15    Rock      Rock       D      +0          +0
```

**Exploitability = 0.010** (best response: always play Paper for expected payoff +0.010/round)

### Rock-biased agent (exploitable) vs Best Response

Agent policy: R=0.70  P=0.15  S=0.15
Opponent policy: R=0.00  P=1.00  S=0.00

```
Round   Agent  Opponent  Result  Payoff  Cumulative
-------------------------------------------------------
    1    Rock     Paper       L      -1          -1
    2    Rock     Paper       L      -1          -2
    3    Rock     Paper       L      -1          -3
    4  Scissors     Paper       W      +1          -2
    5  Scissors     Paper       W      +1          -1
    6   Paper     Paper       D      +0          -1
    7    Rock     Paper       L      -1          -2
    8    Rock     Paper       L      -1          -3
    9    Rock     Paper       L      -1          -4
   10    Rock     Paper       L      -1          -5
   11    Rock     Paper       L      -1          -6
   12    Rock     Paper       L      -1          -7
   13    Rock     Paper       L      -1          -8
   14  Scissors     Paper       W      +1          -7
   15    Rock     Paper       L      -1          -8
```

**Exploitability = 0.550** (best response: always play Paper for expected payoff +0.550/round)

### Cycling agent vs Cycling opponent (self-play failure)

Agent policy: R=0.80  P=0.10  S=0.10
Opponent policy: R=0.10  P=0.80  S=0.10

```
Round   Agent  Opponent  Result  Payoff  Cumulative
-------------------------------------------------------
    1    Rock     Paper       L      -1          -1
    2    Rock     Paper       L      -1          -2
    3   Paper      Rock       W      +1          -1
    4    Rock     Paper       L      -1          -2
    5    Rock     Paper       L      -1          -3
    6    Rock     Paper       L      -1          -4
    7    Rock     Paper       L      -1          -5
    8   Paper     Paper       D      +0          -5
    9    Rock      Rock       D      +0          -5
   10  Scissors     Paper       W      +1          -4
   11    Rock     Paper       L      -1          -5
   12    Rock     Paper       L      -1          -6
   13    Rock     Paper       L      -1          -7
   14   Paper     Paper       D      +0          -7
   15    Rock     Paper       L      -1          -8
```

**Exploitability = 0.700** (best response: always play Paper for expected payoff +0.700/round)
