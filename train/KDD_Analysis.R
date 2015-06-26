# ==============================================================================
# LIBRARIES
# ==============================================================================

library(dplyr)
library(readr)
library(lubridate)
library(magrittr)
library(ggplot2)
library(ROCR)
library(stringr)


# ==============================================================================
# LOADING DATA
# ==============================================================================

enroll_df <- read_csv("/Users/ryo/work/etc/kddcup2015/train/enrollment_train.csv")
log_df <- read_csv("/Users/ryo/work/etc/kddcup2015/train/log_train.csv", col_types = list(time = col_character()))
object_df <- read_csv("/Users/ryo/work/etc/kddcup2015/object.csv")
label_df <- read_csv("/Users/ryo/work/etc/kddcup2015/train/truth_train.csv",
                     col_names = c("enrollment_id", "dropout"))



# Don't have label for enroll id 139669. Remove it from data
log_df %<>% filter(enrollment_id != 139669)
enroll_df %<>% filter(enrollment_id != 139669)

# Add labels to enroll_df
enroll_df <- inner_join(enroll_df, label_df)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Create a function to compute the AUC
calcAUC <- function(predcol, outcol) {
    perf <- performance(prediction(predcol, outcol == 1), "auc")
    as.numeric(perf@y.values)
}

# Create a function to make detail df
create_detail <- function(log_df, object_df) {
    
    # Format time column to make a POSIXct object
    log_df %<>% mutate(time = ymd_hms(gsub("T", " ", time)),
                       event_date = as.Date(time),
                       access_date = event_date)

    log_df[log_df$event != "access",]$access_date <- NA
    
    object_df %<>% mutate(start = ymd_hms(gsub("T", " ", start)),
                          module_release_dt = as.Date(start))
    # Get number of children per module
    object_df$num_children <- sapply(object_df$children,
                                     function(x) length(str_split(x, " ")[[1]]))
    
    # Change spelling error nagivate to navigate!
    log_df[log_df$event == "nagivate",]$event = "navigate"
    
    # Get course start and end date for each course
    log_df %<>%
        group_by(course_id) %>%
        summarise(course_strt_dt = as.Date(min(time)),
                  course_end_dt = as.Date(max(time))) %>%
        inner_join(log_df, .)
    
    # Join log_df to object_df
    detail_df <- left_join(log_df, object_df, by = c("course_id" = "course_id",
                                                     "object" = "module_id"))
}

# Create a function to make some summary features
create_summary <- function(df) {
    summary_df <- 
        df %>%
        group_by(enrollment_id, username, course_id) %>%
        summarise(num_videos = sum(ifelse(event == "video", 1, 0)),
                  num_navigate = sum(ifelse(event == "navigate", 1, 0)),
                  num_access = sum(ifelse(event == "access", 1, 0)),
                  num_problem = sum(ifelse(event == "problem", 1, 0)),
                  num_page_close = sum(ifelse(event == "page_close", 1, 0)),
                  num_discussion = sum(ifelse(event == "discussion", 1, 0)),
                  num_wiki = sum(ifelse(event == "wiki", 1, 0)),
                  num_events = n(),
                  lst_wk_strt_date = max(course_end_dt) - 7,
                  lst_2wk_strt_date = max(course_end_dt) - 14,
                  num_events_lst_wk = sum(
                      ifelse(event_date >= lst_wk_strt_date,
                             1, 0)),
                  num_access_lst_wk = sum(
                      ifelse(event_date >= lst_wk_strt_date & event == "access",
                             1, 0)),
                  num_access_lst2_wk = sum(
                      ifelse(event_date >= lst_2wk_strt_date & event == "access",
                             1, 0)),
                  days_course_strt_access1 = as.numeric(
                      min(access_date, na.rm = T) - max(course_strt_dt)),
                  days_course_end_access_lst = as.numeric(
                      max(course_end_dt) - max(access_date, na.rm = T)),

                  unique_days_accessed = n_distinct(event_date)
        ) %>%
        ungroup
    
    # Get features at module level like the median days between a module release
    # and the first access by user
    summary_df <- 
        detail_df %>%
        group_by(enrollment_id, username, course_id, object) %>%
        summarise(days_acs1_mod_rls = as.numeric(
            min(access_date, na.rm = T) - max(module_release_dt, na.rm = T)),
            days_acslst_mod_rls = as.numeric(
                max(access_date, na.rm = T) - max(module_release_dt,
                                                  na.rm = T))) %>%
        ungroup %>%
        group_by(enrollment_id, username, course_id) %>%
        summarise(median_days_acs1_mod_rls = median(days_acs1_mod_rls,
                                                    na.rm = T),
                  median_days_acslst_mod_rls = median(days_acslst_mod_rls,
                                                      na.rm = T)) %>%
        ungroup %>%
        left_join(summary_df, .)
    
    # Replace NAs by 9999
    summary_df[is.na(summary_df)] <- 9999
    
    summary_df
}


# ==============================================================================
# TRAIN TEST SPLIT
# ==============================================================================

# Create detail df
detail_df <- create_detail(log_df, object_df)

# Free up some memory by removing unneeded objects
rm(log_df)
gc()

# Create summary df
summary_df <- create_summary(detail_df)
summary_df %<>% inner_join(label_df)

# Split into test and train
set.seed(729375)
summary_df$rgroup <- runif(nrow(summary_df))
train <- summary_df %>% filter(rgroup <= 0.8)
test <- summary_df %>% filter(rgroup > 0.8)
train$rgroup <- NULL
test$rgroup <- NULL


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

# Let's see the trend of dropouts with unique number of days MOOC was accessed
ggplot(train, aes(x = unique_days_accessed, y = as.numeric(dropout))) +
    geom_point(position = position_jitter(w = 0.05, h = 0.05)) +
    geom_smooth() +
    xlab("Number of unique days MOOC was accessed") +
    ylab("Dropout") +
    ggtitle("Trend of dropouts with unique number of days accessed")


# ==============================================================================
# MODELS
# ==============================================================================

# Let's fit a decision tree
library(rpart)
library(rpart.plot)

train %<>% select(num_videos:num_events, num_events_lst_wk:dropout)

tree.model <- rpart(as.factor(dropout) ~ ., data = train,
                    control = rpart.control(maxdepth = 7))
prp(tree.model)

pred <- predict(tree.model, newdata = test)
pred <- ifelse(pred[, 2] >= 0.5, 1, 0)
table(pred, test$dropout)
calcAUC(pred, test$dropout)


# ==============================================================================
# SUBMISSIONS
# ==============================================================================

# Load actual test data
enroll_test_df <- read_csv("../test/enrollment_test.csv")
log_test_df <- read_csv("../test/log_test.csv",
                        col_types = list(time = col_character()))

# Create test detail df
detail_test_df <- create_detail(log_test_df, object_df)

# Free up some memory by removing unneeded objects
rm(log_test_df, object_df)
gc()

# Create test summary df
summary_test_df <- create_summary(detail_test_df)
summary_test_df %<>% left_join(enroll_test_df, .)

# Predict
pred <- predict(tree.model, newdata = summary_test_df)
pred <- ifelse(pred[, 2] >= 0.5, 1, 0)
submit_df <- data.frame(enroll_id = enroll_test_df$enrollment_id, 
                        prediction = pred)

# Write results to file
write_csv(submit_df, "/Users/ryo/work/etc/kddcup2015/R_hoge_result.csv",
          col_names = F)
