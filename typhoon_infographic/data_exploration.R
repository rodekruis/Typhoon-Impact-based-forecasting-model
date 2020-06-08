library(readr)
library(readxl)
library(ggplot2)

age_pop <- read_xlsx("C:/Users/letic/Desktop/Typhoon-Impact-based-forecasting-model/data/Indicators_4_population_by_age.xlsx")

age_pop$perc_u5 <- as.numeric(age_pop$perc_u5)

age_pop2 <- na.omit(age_pop)

sum(age_pop2$perc_u5)



#############################################################################

a <- c("children", "adults", "elderly")
b <- c(298836, 611954, 50210)

agepop <- data.frame(a, b)



#piechart
    ggplot(agepop, aes(x= "", y = b, fill= a))+
       geom_bar(width = 1, stat = "identity") +
       coord_polar("y", start = 0) +
       theme_void() +
      theme(
        legend.position = "none")  +
       scale_fill_manual(values = c("grey83", "red2", "grey50"))


  ggplot(agepop, aes(x= "", y = b, fill= a))+
  geom_bar(width = 1, stat = "identity") +
  coord_polar("y", start = 0) +
  theme_void() +
  theme(
    legend.position = "none")  +
  scale_fill_manual(values = c("red2", "grey83", "grey50"))


  ggplot(agepop, aes(x= "", y = b, fill= a))+
  geom_bar(width = 1, stat = "identity") +
  coord_polar("y", start = 0) +
  theme_void() +
  theme(
    legend.position = "none")  +
  scale_fill_manual(values = c("grey83", "grey50", "red2"))



