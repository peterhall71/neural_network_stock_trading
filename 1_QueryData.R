# load libraries
library(RMySQL)

# connect to mysql database
con <- dbConnect(MySQL(), 
		user = 'phall', 
		password = 'raspberry', 
		dbname = 'dhData', 
		host = '192.168.1.2'
)

# query database for stock data
query <- "SELECT * FROM dhData.ohlc_1min_interval"
db_results <- dbGetQuery(con, query)

# disconnect from database
dbDisconnect(con)
