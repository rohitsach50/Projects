### SQL file contains schema of mysql database 

### data_extracting.py file downloads csv file related to air quality from data.gov.in and renames the csv file to datetime of file creation.

### CSV_TO_MYSQL.py file extract data from csv file and transforms it before loading it to mysql database on AWS RDS service.

### cron is used for scheduling tasks of running data_extracting.py file at minute 1 past every hour and then uploading the csv file to aws s3.
