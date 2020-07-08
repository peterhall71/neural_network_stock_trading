#### NEED TO CHANGE TO 44 COLUMNS

# load libraries
library(data.table)

# prepare dataframe
dfprep <- db_results

# drop unneeded columns
ohlc <- dfprep[,1:4]

# transpose dataframe and convert to vector
# structure created is: open1, high1, low1, close1, open2, high2, low2, close2, etc...
ohlcVec <- as.vector(t(ohlc))

# create subsetting function
subsetVector <- function(y, i) {
	last <- i + 44
	return(y[i:last])
}

# create dataframe
ohlcList <- lapply(seq_along(ohlcVec), subsetVector, y = ohlcVec)
ohlcDF <- as.data.frame(t(as.data.frame(ohlcList)))

# clean up dataframe
rownames(ohlcDF) <- NULL
ohlcDF <- na.omit(ohlcDF)

# normalize, divide rows by mean
ohlcNorm <- ohlcDF/rowMeans(ohlcDF)

# code training data and drop unneeded column
key <- apply(ohlcNorm, 1, function(x) {ifelse(x[45] > x[41], 1, 0)})
trainingData <- data.frame(ohlcNorm, key, row.names = NULL)
trainingData <- trainingData[ ,-c(42:45)]

# write training data to csv
write.csv(trainingData,"trainingData.csv", row.names = FALSE)
