Place the public sample file here as `total_pat.csv`.

Required columns:

- `reg_num`: patent identifier
- `pat_year`: integer publication/registration year
- `patent_title`: title
- `patent_abstract`: abstract

The default pipeline uses patents from 2010 through 2021. Years 2010–2019 are used to fit semantic clusters, and 2020–2021 are assigned to those clusters.
