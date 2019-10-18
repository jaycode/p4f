## Self-Serve Data to Quantopian Research
First, we download the data. In this case, we want to get GDP per-capita data, which we can download from the [World Bank website](https://data.worldbank.org/indicator/ny.gdp.pcap.cd).

Take a quick look on this data, and you'll see the following section:

![csv_headers](https://github.com/jaycode/p4f/raw/master/lessons/self-serve_data/csv_headers.png)

See that only row 5 contains the actual field names.

Pandas' `read_csv()` function omits the empty lines be
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjc4MjIxNDYsLTkxMTk0ODUyLC0xMjA3MD
kyNzY0XX0=
-->