Explanation of the methods to compute the statistics in this directory:

Age groups are named as follows in all files:

| age_code      | age group   |
| ------------- |:-----------|
|0 | 0-9 years old
|1 |10-19 years old
|2 |20-29 years old
|3 |30-39 years old
|4 |40-49 years old
|5 |50-59 years old
|6 |60-69 years old
|7 |70-79 years old
|8 |80+ years old

# asymptomatic_fraction.csv
**The fraction of asymptomatic cases for each age group.** In the data, there is a column named "no symptoms", with values 1 or 0. For four patient it was 4 and we discarded these records in the analysis. We expect these numbers to underestimate the real asymptomatic fractions, because we are likely not aware of most of the asymptomatic cases.

* `Count:` the total number of patients in the age group (not only asymptomatic).
* `Mean:` the fraction of asymptomatic within an age group. 
* `Binomial std`: the standard deviation we give to our mean estimate. We assumed that the mean is distributed according to a binomial distribution: mean = Bin(n=count, p=mean) / count. hence: binomial std = sqrt(count * mean * (1-mean)) / count.

# hospitalization_rate_of_symptomatic.csv
**The fraction of hospitalization out of all symptomatic cases for each age group.** In the data, there is a column named `no symptoms`, with values 1 or 0. For four patient it was 4 and we discarded these records in the analysis We considered all subjects for which `no symptoms==0` as symptomatic.

* `count` is the total number of symptomatic in an age group.
* `mean` is the fraction of hospitalized (people whose hospital is not `NaN`) out of the symptomatic group.
* `binomial std` is the standard deviation we give to our mean estimate. We assumed that mean is distributed according to a binomial distribution: mean = Bin(n=count, p=mean) / count. Hence: binomial std = sqrt(count * mean * (1-mean)) / count.


# infection_to_hospitalization.csv
The time in days from first positive test to hospital admission, for all patients for which both times exist. We excluded 524 patients for which time of first positive test was after hospital admission. These outliers suggest that our calculations may underestimate the time from infection to hospitalization.

# infection_to_death.csv
**The time in days from first positive test to death, for all patients for which both times exist.** We excluded 5 patients for which the time of first positive test was after death. These may have been errors in the data, or real data of subjects who were tested after they died. These outliers suggest that our calculations may underestimate the time from infection to death.

# death_rate_of_hospitalized.csv
**The fraction of deaths out of all hospitalized cases for each age group.**

* `count` is the total number of hospitalized in an age group.
* `mean`  is the fraction of deaths out of the hospitalized. 
* `binomial std` is the standard deviation we give to our mean estimate. We assumed that mean is distributed according to a binomial distribution: mean = Bin(n=count, p=mean) / count. Hence: binomial std = sqrt(count * mean * (1-mean)) / count. Note that binomial std can underestimate the real variance since it assumes we get the mean right. This is especially important for cases where mean=0, such as case mortality of young people.


# admission_to_discharge.csv

**The time in days from admission to discharge**, for all patients for which both times exist.

# cumulative_cases.csv

**The cumulative number of covid19 cases up to a given date, per age group.** Columns are age groups as defined above. Rows are dates, date = 0 is 14/2, date = 1 is 15/2 etc. For each date we document the number of subjects whose first positive test date is before (or equal) this date.






 

