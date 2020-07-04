# load libraries
library(RMySQL)

# connect to mysql database
con <- dbConnect(MySQL(), 
		user = 'user', 
		password = 'password', 
		dbname = 'dbname', 
		host = 'host'
)

# query database for stock data
query <- "SELECT * FROM dhData.ohlc_1min_interval"
db_results <- dbGetQuery(con, query)

# disconnect from database
dbDisconnect(con)
