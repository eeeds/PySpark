"""
SQL in a nutshell
As you move forward, it will help to have a basic understanding of SQL. A more in depth look can be found here.

A SQL query returns a table derived from one or more tables contained in a database.

Every SQL query is made up of commands that tell the database what you want to do with the data. The two commands that every query has to contain are SELECT and FROM.

The SELECT command is followed by the columns you want in the resulting table.

The FROM command is followed by the name of the table that contains those columns. The minimal SQL query is:

SELECT * FROM my_table;
The * selects all columns, so this returns the entire table named my_table.

Similar to .withColumn(), you can do column-wise computations within a SELECT statement. For example,

SELECT origin, dest, air_time / 60 FROM flights;
returns a table with the origin, destination, and duration in hours for each flight.

Another commonly used command is WHERE. This command filters the rows of the table based on some logical condition you specify. The resulting table contains the rows where your condition is true. For example, if you had a table of students and grades you could do:

SELECT * FROM students
WHERE grade = 'A';
to select all the columns and the rows containing information about students who got As.

Which of the following queries returns a table of tail numbers and destinations for flights that lasted more than 10 hours?
SELECT dest, tail_num FROM flights WHERE air_time > 10;

(Answer)SELECT dest, tail_num FROM flights WHERE air_time > 600;

SELECT * FROM flights WHERE air_time > 600;
"""


"""
SQL in a nutshell (2)
Another common database task is aggregation. That is, reducing your data by breaking it into chunks and summarizing each chunk.

This is done in SQL using the GROUP BY command. This command breaks your data into groups and applies a function from your SELECT statement to each group.

For example, if you wanted to count the number of flights from each of two origin destinations, you could use the query

SELECT COUNT(*) FROM flights
GROUP BY origin;
GROUP BY origin tells SQL that you want the output to have a row for each unique value of the origin column. The SELECT statement selects the values you want to populate each of the columns. Here, we want to COUNT() every row in each of the groups.

It's possible to GROUP BY more than one column. When you do this, the resulting table has a row for every combination of the unique values in each column. The following query counts the number of flights from SEA and PDX to every destination airport:

SELECT origin, dest, COUNT(*) FROM flights
GROUP BY origin, dest;
The output will have a row for every combination of the values in origin and dest (i.e. a row listing each origin and destination that a flight flew to). There will also be a column with the COUNT() of all the rows in each group.

Remember, a more in depth look at SQL can be found here.

What information would this query get? Remember the flights table holds information about flights that departed PDX and SEA in 2014 and 2015. Note that AVG() function gets the average value of a column!

SELECT AVG(air_time) / 60 FROM flights
GROUP BY origin, carrier;

(Answer) The average length of each airline's flights from SEA and from PDX in hours.

The average length of each flight.

The average length of each airline's flights.
"""



