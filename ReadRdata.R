rm(list = ls())
# mathc with picture time points
library(tidyverse)
# import CGM data
load(file = '../CGM_DATA/CGM01202022.RData')

picture_tm <- rio::import(file = '/Volumes/Research/Metabolomics/Diet_challenge/data/CGM/images-v1-copsac-approxtimetaken.xlsx')

CGM <- X %>% 
  filter(!str_detect(name,'Pia')) %>% 
  filter(!str_detect(name,'Hans')) %>% 
  filter(!str_detect(name,'Axel')) %>% 
  filter(!str_detect(name,'uvist270717')) %>%
  mutate(copsacno = name %>% parse_number(), 
         glucose_mmol_L = Historic.Glucose..mmol.L., 
         Time2 = Time %>% as.POSIXct("%Y/%m/%d %H:%M:%OS")) %>% 
  select(copsacno,name,Time,Time2,glucose_mmol_L)

picture_tm <- picture_tm %>% 
  mutate(copsacno = Folder %>% tolower %>% str_remove('cp-') %>% as.numeric())
CGM %>% str
picture_tm %>% str
cpno <- picture_tm[1,]
# find closest point

picture_tm <- picture_tm[picture_tm$copsacno %in% CGM$copsacno,]
picture_tm <- picture_tm %>% filter(!is.na(copsacno))
picture_tm %>% dim
GLUC <- c()
for (i in 1:dim(picture_tm)[1]){
  # i <- 1
  # print(i)
  cpno <- picture_tm$copsacno[i]
  cgm <- CGM %>% filter(copsacno==cpno) %>% 
    arrange(Time2)
  dt <- difftime(cgm$Time2 ,picture_tm$TimeModifiedImage[i],units = 'mins')
  idmid <- which.min(abs(dt))
  # r <- ifelse(length(idmid)==0,   
  # c(cpno,rep(NA,18)),
  # icc <- (idmid - 4):(idmid+4)
  icc <- (idmid - 8):(idmid+4)
  icc <- icc[icc>0]
  cgm2 <- cgm[icc,]
  timerelative2intake <- round(as.numeric(difftime(cgm2$Time2,picture_tm$TimeModifiedImage[i],units = 'mins')))
  glucose <- cgm2$glucose_mmol_L
  r <- c(cpno,timerelative2intake,glucose)
  r <- if(length(r)<27) {r = c(cpno,rep(NA,26))} else {r = c(cpno,timerelative2intake,glucose)}
  # )
  GLUC <- rbind(GLUC,r)
}

GLUC <- data.frame(GLUC)
colnames(GLUC)<- c('copsacno',paste(rep('reltime_min',13),-8:4, sep = '_'),
                   paste(rep('glucose_mmol_L_pt',13),-8:4))
picture_with_glucose_timemodified <- picture_tm %>% cbind(GLUC[,-1])


GLUC <- c()
picture_tm2 <- picture_tm %>% filter(!is.na(TimePhoneImage))
for (i in 1:dim(picture_tm2)[1]){
  # print(i)
  cpno <- picture_tm2$copsacno[i]
  cgm <- CGM %>% filter(copsacno==cpno) %>% 
    arrange(Time2)
  dt <- difftime(cgm$Time2 ,picture_tm2$TimePhoneImage[i],units = 'mins')
  idmid <- which.min(abs(dt))
  # r <- ifelse(length(idmid)==0, 
  # c(cpno,rep(NA,18)),
  # icc <- (idmid - 4):(idmid+4)
  icc <- (idmid - 8):(idmid+4)
  icc <- icc[icc>0]
  cgm2 <- cgm[icc,]
  timerelative2intake <- round(as.numeric(difftime(cgm2$Time2,picture_tm2$TimePhoneImage[i],units = 'mins')))
  glucose <- cgm2$glucose_mmol_L
  r <- c(cpno,timerelative2intake,glucose)
  r <- if(length(r)<27) {r = c(cpno,rep(NA,26))} else {r = c(cpno,timerelative2intake,glucose)}
  # )
  GLUC <- rbind(GLUC,r)
}

GLUC <- data.frame(GLUC)
colnames(GLUC)<- c('copsacno',paste(rep('reltime_min',13),-8:4, sep = '_'),
                   paste(rep('glucose_mmol_L_pt',13),-8:4))
picture_with_glucose_timephone <- picture_tm2 %>% cbind(GLUC[,-1])



GLUC <- c()
picture_tm2 <- picture_tm %>% filter(!is.na(TimeFilenameImage))
for (i in 1:dim(picture_tm2)[1]){
  # print(i)
  cpno <- picture_tm2$copsacno[i]
  cgm <- CGM %>% filter(copsacno==cpno) %>% 
    arrange(Time2)
  dt <- difftime(cgm$Time2 ,picture_tm2$TimeFilenameImage[i],units = 'mins')
  idmid <- which.min(abs(dt))
  # r <- ifelse(length(idmid)==0, 
  # c(cpno,rep(NA,18)),
  # icc <- (idmid - 4):(idmid+4)
  icc <- (idmid - 8):(idmid+4)
  icc <- icc[icc>0]
  cgm2 <- cgm[icc,]
  timerelative2intake <- round(as.numeric(difftime(cgm2$Time2,picture_tm2$TimeFilenameImage[i],units = 'mins')))
  glucose <- cgm2$glucose_mmol_L
  r <- c(cpno,timerelative2intake,glucose)
  r <- if(length(r)<27) {r = c(cpno,rep(NA,26))} else {r = c(cpno,timerelative2intake,glucose)}
  # )
  GLUC <- rbind(GLUC,r)
}

GLUC <- data.frame(GLUC)
colnames(GLUC)<- c('copsacno',paste(rep('reltime_min',13),-8:4, sep = '_'),
                   paste(rep('glucose_mmol_L_pt',13),-8:4))
picture_with_glucose_timefile <- picture_tm2 %>% cbind(GLUC[,-1])

list_of_datasets <-
  list("timefile" = picture_with_glucose_timefile, 
       "timemodified" = picture_with_glucose_timemodified, 
       "timephone" = picture_with_glucose_timephone)

openxlsx::write.xlsx(list_of_datasets, '/Volumes/Research/Metabolomics/Diet_challenge/data/CGM/images-v1-copsac-approxtimetaken_addedglucose_v2.xlsx', overwrite = T)
