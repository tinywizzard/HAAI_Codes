data <- read.csv("C:\\Users\\kajal\\Desktop\\Python codes\\R\\SM_data.csv")
head(data)
dim(data)
# Mean
mean(data$H)
mean(data$PL)
mean(data$AL)
mean(data$FL)
mean(data$MF == "M")

# Median
median(data$H)
mean(data$H)

# Variance
var(data$H)		# Unbiased
sd(data$H)
sum((data$H - mean(data$H))^2)
sum((data$H - mean(data$H))^2) / (length(data$H))		# Biased
sum((data$H - mean(data$H))^2) / (length(data$H) - 1)	# Unbiased

# Range

range(data$H)

# Summary
summary(data$H)
summary(data)

# Fitting Normal distribution to data$PL
hist(data$PL)
hist(data$PL, breaks = 20)
hist(data$PL, breaks = 20, probability = TRUE)
m <- mean(data$PL)
v <- var(data$PL)

s <- seq(min(data$PL), max(data$PL), by = 0.05)

# Normal distribution
lines(dnorm(s, mean = m, sd = sqrt(v)) ~ s, col = 2, lwd = 2, lty = 2)
# Non-parametric density estimation or Kernel density estimation (KDE)
lines(density(data$PL), col = 3, lwd = 3, lty = 3)

meant <- t.test(data$PL, conf.level = 0.95)
cat("95% CI  of the  mean of palm length is  (", meant$conf.int[1:2], ")\n")

library("TeachingDemos")
varchisq <- sigma.test(data$PL, sigma = 3, alternative = "two.sided", conf.level = 0.95)
cat("95% CI  of the var of palm length is  (", varchisq$conf.int[1:2], ")\n")

#===================================

cov_mat <- cov(data[1:4])
print(cov_mat)

cor(data[1:4])

eg_cov <- eigen(cov_mat)
print(eg_cov$values)

reg1 <- lm(data$H ~ data$PL + data$AL + data$FL)
reg2 <- lm(data$H ~ data$PL)

plot(data$H ~ data$PL)
#ss <- seq(min(data$PL), max(data$PL), by=0.5)
abline(a = reg2$coefficients[1], b =  reg2$coefficients[2], col = 2, lwd = 4)
