
# load libraries
library(data.table)

# prepare dataframe
dfprep <- db_results
db_results$ticker <- as.factor(db_results$ticker)

# subset dataframe, drop unneeded columns
aapl <- dfprep[dfprep$ticker == "aapl", ]
aaplOHLC <- aapl[,1:4]

# transpose dataframe and convert to vector
# structure created is: open1, high1, low1, close1, open2, high2, low2, close2, etc...
aaplVec <- as.vector(t(aaplOHLC))

# create subsetting function
subsetVector <- function(y, i) {
	last <- i + 44
	return(y[i:last])
}

# create dataframe
aapl44list <- lapply(seq_along(aaplVec), subsetVector, y = aaplVec)
aapl44df <- as.data.frame(t(as.data.frame(aapl44list)))

# clean up dataframe
rownames(aapl44df) <- NULL
aapl44df <- na.omit(aapl44df)

# normalize, divide rows by mean
aapl44norm <- aapl44df/rowMeans(aapl44df)

# code training data and drop unneeded column
key <- apply(aapl44norm, 1, function(x) {ifelse(x[45] > x[41], 1, 0)})
trainingData <- data.frame(aapl44norm, key, row.names = NULL)
trainingData <- trainingData[ ,-c(42:45)]

