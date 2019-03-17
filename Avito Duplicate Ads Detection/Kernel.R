

rm(list = ls())

## Load Package
library(tidyverse)
library(data.table)
library(geosphere)
library(stringdist)
library(xgboost)
library(randomForest)
set.seed(0)

## Load csv
setwd("E:/Avito Duplicate Ads Detection/HW4")

Category=fread("Category.csv")
Location=fread("Location.csv")
ItemPairs_train=fread("ItemPairs_train.csv")
ItemPairs_test=fread("ItemPairs_test.csv")
ItemInfo_train=read_csv("ItemInfo_train.csv")
ItemInfo_test=read_csv("ItemInfo_test.csv")

ItemInfo_test=data.table(ItemInfo_test)
ItemInfo_train=data.table(ItemInfo_train)

## Data wraggling
#Location
ItemInfo_train=ItemInfo_train %>% 
  left_join(Location)

ItemInfo_test=ItemInfo_test %>% 
  left_join(Location)

#Train
train=ItemPairs_train %>% 
  left_join(ItemInfo_train,by = c("itemID_1" = "itemID"))
colnames(train)[5:15]=paste0(colnames(train)[5:15],"_1")

train=train %>% 
  left_join(ItemInfo_train,by = c("itemID_2" = "itemID"))
colnames(train)[16:26]=paste0(colnames(train)[16:26],"_2")

#Test
test=ItemPairs_test %>% 
  left_join(ItemInfo_test,by = c("itemID_1" = "itemID"))
colnames(test)[4:14]=paste0(colnames(test)[4:14],"_1")

test=test %>% 
  left_join(ItemInfo_test,by = c("itemID_2" = "itemID"))
colnames(test)[15:25]=paste0(colnames(test)[15:25],"_2")

## Remove unnecessary dataset
rm(list=c("Category", "Location", "ItemPairs_train","ItemPairs_test",
          "ItemInfo_train","ItemInfo_test"))

## Create features
test=data.table(test)
train=data.table(train)

same_or_not=function(x,y){
  ifelse(is.na(x)==FALSE & is.na(y)==FALSE, ifelse(x==y,1,-1),0)
}
# same is 1, not same is -1, have NA is 0

#Add title-description distance
add_t_d_dist=function(Item){
  Item[,':='(
    title_description_dist1_1=stringdist(Item$title_1, Item$description_1, method = "jw"),
    title_description_dist1_2=stringdist(Item$title_2, Item$description_2, method = "jw"),
    title_description_dist2_1=stringdist(Item$title_1, Item$description_1, method = "cosine"),
    title_description_dist2_2=stringdist(Item$title_2, Item$description_2, method = "cosine"),
    title_description_dist3_1=stringdist(Item$title_1, Item$description_1, method = "jaccard"),
    title_description_dist3_2=stringdist(Item$title_2, Item$description_2, method = "jaccard")
  )]
  
  Item[,':='(
    title_description_dist1_1=ifelse(is.na(title_description_dist1_1)==TRUE,0,title_description_dist1_1),
    title_description_dist1_2=ifelse(is.na(title_description_dist1_2)==TRUE,0,title_description_dist1_2),
    title_description_dist2_1=ifelse(is.na(title_description_dist2_1)==TRUE,0,title_description_dist2_1),
    title_description_dist2_2=ifelse(is.na(title_description_dist2_2)==TRUE,0,title_description_dist2_2),
    title_description_dist3_1=ifelse(is.na(title_description_dist3_1)==TRUE,0,title_description_dist3_1),
    title_description_dist3_2=ifelse(is.na(title_description_dist3_2)==TRUE,0,title_description_dist3_2)
  )]
}

#Add num and nchar features
add_num_features=function(Item){
  Item[,':='(
    num_images_1=ifelse(is.na(images_array_1),0,str_count(images_array_1,",")+1),
    num_images_2=ifelse(is.na(images_array_2),0,str_count(images_array_2,",")+1),
    num_attrs_1=ifelse(is.na(attrsJSON_1),0,str_count(attrsJSON_1,",")+1),
    num_attrs_2=ifelse(is.na(attrsJSON_2),0,str_count(attrsJSON_2,",")+1),
    
    nchartitle_1=ifelse(is.na(title_1)==TRUE,0,nchar(title_1)),
    nchartitle_2=ifelse(is.na(title_2)==TRUE,0,nchar(title_2)),
    nchardescription_1=ifelse(is.na(description_1)==TRUE,0,nchar(description_1)),
    nchardescription_2=ifelse(is.na(description_2)==TRUE,0,nchar(description_2)),
    ncharattrsJSON_1=ifelse(is.na(attrsJSON_1)==TRUE,0,nchar(attrsJSON_1)),
    ncharattrsJSON_2=ifelse(is.na(attrsJSON_2)==TRUE,0,nchar(attrsJSON_2))
    
  )]
}


#Add match features
add_match_features=function(Item){
  Item[,':='(
    location_match=same_or_not(locationID_1, locationID_2),
    region_match=same_or_not(regionID_1, regionID_2),
    metro_match=same_or_not(metroID_1, metroID_2),
    price_match=same_or_not(price_1, price_2),
    num_images_match=same_or_not(num_images_1,num_images_2),
    num_attrs_match=same_or_not(num_attrs_1,num_attrs_2),
    title_nchar_match=same_or_not(nchartitle_1,nchartitle_2),
    description_nchar_match=same_or_not(nchardescription_1,nchardescription_2),
    attrsJSON_nchar_match=same_or_not(ncharattrsJSON_1,ncharattrsJSON_2)
    
  )]
}

#Create compare model features
create_features=function(Item){
  Item[,':='(
    #location
    same_locationID=ifelse(location_match==1,locationID_1,0),
    locationID_1=NULL,
    locationID_2=NULL,
    
    #region
    same_regionID=ifelse(region_match==1,regionID_1,0),
    regionID_1=NULL,
    regionID_2=NULL,
    
    #metro
    same_metroID=ifelse(metro_match==1,metroID_1,0),
    metroID_1=NULL,
    metroID_2=NULL,
    
    categoryID_1 = NULL,
    categoryID_2 = NULL,
    
    #price
    same_price=ifelse(price_match==1,price_1,0),
    price_diff=ifelse(price_match==0,0,abs(price_1-price_2)),
    price_ratio=ifelse(price_match==0,0,pmin(price_1,price_2)/pmax(price_1,price_2)),
    price_1=NULL,
    price_2=NULL,
    
    #num of images
    same_num_images=ifelse(num_images_match==1,num_images_1,0),
    num_images_diff=abs(num_images_1-num_images_2),
    num_images_ratio=ifelse(num_images_1==0&num_images_2==0,0,
                            pmin(num_images_1,num_images_2)/pmax(num_images_1,num_images_2)),
    images_array_1=NULL,
    images_array_2=NULL,
    num_images_1=NULL,
    num_images_2=NULL,
    
    #num of attrs
    same_num_attrs=ifelse(num_attrs_match==1,num_attrs_1,0),
    num_attrs_diff=abs(num_attrs_1-num_attrs_2),
    num_attrs_ratio=ifelse(num_attrs_1==0&num_attrs_2==0,0,
                           pmin(num_attrs_1,num_attrs_2)/pmax(num_attrs_1,num_attrs_2)),
    num_attrs_1=NULL,
    num_attrs_2=NULL,
    
    #num of char of title
    same_title_nchar=ifelse(title_nchar_match==1,nchartitle_1,0),
    title_nchar_diff=abs(nchartitle_1-nchartitle_2),
    title_nchar_ratio=ifelse(nchartitle_1==0&nchartitle_2==0,0,
                             pmin(nchartitle_1,nchartitle_2)/pmax(nchartitle_1,nchartitle_2)),
    nchartitle_1=NULL,
    nchartitle_2=NULL,
    
    #num of char of description
    same_description_nchar=ifelse(description_nchar_match==1,nchardescription_1,0),
    description_nchar_diff=abs(nchardescription_1-nchardescription_2),
    description_nchar_ratio=ifelse(nchardescription_1==0&nchardescription_2==0,0,
                                   pmin(nchardescription_1,nchardescription_2)/pmax(nchardescription_1,nchardescription_2)),
    nchardescription_1=NULL,
    nchardescription_2=NULL,
    
    #num of char of attrsJSON
    same_attrsJSON_nchar=ifelse(attrsJSON_nchar_match==1,ncharattrsJSON_1,0),
    attrsJSON_nchar_diff=abs(ncharattrsJSON_1-ncharattrsJSON_2),
    attrsJSON_nchar_ratio=ifelse(ncharattrsJSON_1==0&ncharattrsJSON_2==0,0,
                                 pmin(ncharattrsJSON_1,ncharattrsJSON_2)/pmax(ncharattrsJSON_1,ncharattrsJSON_2)),
    ncharattrsJSON_1=NULL,
    ncharattrsJSON_2=NULL,
    
    #distance of title
    title_dist1=stringdist(title_1, title_2, method = "jw"),
    title_dist2=stringdist(title_1, title_2,method = "cosine"),
    title_dist3=stringdist(title_1, title_2,method = "lv"),
    title_dist4=stringdist(title_1, title_2,method = "jaccard"),
    title_1=NULL,
    title_2=NULL,
    
    #distance of description
    description_dist1=stringdist(description_1, description_2, method = "jw"),
    description_dist2=stringdist(description_1, description_2,method = "cosine"),
    description_dist3=stringdist(description_1, description_2,method = "lv"),
    description_dist4=stringdist(description_1, description_2,method = "jaccard"),
    description_1=NULL,
    description_2=NULL,
    
    #distance of attrsJSON
    attrsJSON_dist1=stringdist(attrsJSON_1, attrsJSON_2, method = "jw"),
    attrsJSON_dist2=stringdist(attrsJSON_1, attrsJSON_2,method = "cosine"),
    attrsJSON_dist3=stringdist(attrsJSON_1, attrsJSON_2,method = "lv"),
    attrsJSON_dist4=stringdist(attrsJSON_1, attrsJSON_2,method = "jaccard"),
    attrsJSON_1=NULL,
    attrsJSON_2=NULL,
    
    #title description dist compare
    title_description_dist1_diff=abs(title_description_dist1_1-title_description_dist1_2),
    title_description_dist1_ratio=ifelse(title_description_dist1_1==0&title_description_dist1_2==0,0,
                                         pmin(title_description_dist1_1,title_description_dist1_2)/pmax(title_description_dist1_1,title_description_dist1_2)),
    title_description_dist2_diff=abs(title_description_dist2_1-title_description_dist2_2),
    title_description_dist2_ratio=ifelse(title_description_dist2_1==0&title_description_dist2_2==0,0,
                                         pmin(title_description_dist2_1,title_description_dist2_2)/pmax(title_description_dist2_1,title_description_dist2_2)),
    title_description_dist3_diff=abs(title_description_dist3_1-title_description_dist3_2),
    title_description_dist3_ratio=ifelse(title_description_dist3_1==0&title_description_dist3_2==0,0,
                                         pmin(title_description_dist3_1,title_description_dist3_2)/pmax(title_description_dist3_1,title_description_dist3_2)),
    
    
    #geospatial distance
    distance=distHaversine(cbind(lon_1,lat_1),cbind(lon_2,lat_2)),
    lat_1=NULL,
    lat_2=NULL,
    lon_1=NULL,
    lon_2=NULL,
    
    
    itemID_1=NULL,
    itemID_2=NULL
  )]
  
  Item[,':='(
    title_dist1=ifelse(is.na(title_dist1),0,title_dist1),
    title_dist2=ifelse(is.na(title_dist2),0,title_dist2),
    title_dist3=ifelse(is.na(title_dist3),0,title_dist3),
    title_dist4=ifelse(is.na(title_dist4),0,title_dist4),
    
    description_dist1=ifelse(is.na(description_dist1),0,description_dist1),
    description_dist2=ifelse(is.na(description_dist2),0,description_dist2),
    description_dist3=ifelse(is.na(description_dist3),0,description_dist3),
    description_dist4=ifelse(is.na(description_dist4),0,description_dist4),
    
    attrsJSON_dist1=ifelse(is.na(attrsJSON_dist1),0,attrsJSON_dist1),
    attrsJSON_dist2=ifelse(is.na(attrsJSON_dist2),0,attrsJSON_dist2),
    attrsJSON_dist3=ifelse(is.na(attrsJSON_dist3),0,attrsJSON_dist3),
    attrsJSON_dist4=ifelse(is.na(attrsJSON_dist4),0,attrsJSON_dist4)
  )]
}

test=add_t_d_dist(test)
train=add_t_d_dist(train)

test=add_num_features(test)
train=add_num_features(train)

test=add_match_features(test)
train=add_match_features(train)

test=create_features(test)
train=create_features(train)


test = data.frame(test)
train = data.frame(train)

modelVars = names(train)[which(!(names(train) %in% c("isDuplicate", "generationMethod", "foldId")))]

maxTrees = 130 #95
shrinkage = 0.08 #0.09
gamma = 1
depth = 13 #13
minChildWeight = 38
colSample = 0.4
subSample = 0.37
earlyStopRound = 4


set.seed(0)

# Matrix
dtrain = xgb.DMatrix(as.matrix(train[, modelVars]), label=train$isDuplicate)
dtest = xgb.DMatrix(as.matrix(test[, modelVars]))

xgbResult = xgboost(params=list(max_depth=depth,
                                eta=shrinkage,
                                gamma=gamma,
                                colsample_bytree=colSample,
                                min_child_weight=minChildWeight),
                    data=dtrain,
                    nrounds=maxTrees,
                    objective="binary:logistic",
                    eval_metric="auc")

testPreds = predict(xgbResult, dtest)

submission=data.frame(id=test$id,probability=testPreds)
submission$id=as.integer(submission$id)
write.csv(submission, file="submission.csv",row.names=FALSE)
