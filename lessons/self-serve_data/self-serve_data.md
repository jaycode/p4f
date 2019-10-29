## Self-Serve Data to Quantopian Research

### Problem: You need pricing data for a quick research, but do not want to subscribe to a pricing data provider.

We can perform our analysis in Quantopian in this case.

First, we download the data. In this case, we want to get GDP per-capita data, which we can download from the [World Bank website](https://data.worldbank.org/indicator/ny.gdp.pcap.cd).

Take a quick look on this data, and you'll see the following section:

![csv_headers](https://github.com/jaycode/p4f/raw/master/lessons/self-serve_data/csv_headers.png)

See that row 5 contains the actual field names. The rows before that are not useful in our analysis.

Pandas' `read_csv()` function omits empty lines before the actual header, and it has a parameter to manually omit the rest of the headers. In this case, `header=2` will omit the first and third rows and will correctly use the fifth row as the field names.

To check if we correctly extracted the field names, you may run something like `print(data.info())`


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1OTYwODY2NjksNTYwMTIwNDAwLC05Mj
g0NTU2MTQsLTkxMTk0ODUyLC0xMjA3MDkyNzY0XX0=
-->