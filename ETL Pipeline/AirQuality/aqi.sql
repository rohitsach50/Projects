CREATE SCHEMA `test_db` DEFAULT CHARACTER SET utf8mb4 ;

CREATE TABLE `test_db`.`countries` (
  `idCountry` INT NOT NULL AUTO_INCREMENT,
  `Country_Name` VARCHAR(45) NOT NULL,
  UNIQUE INDEX `idCountry_UNIQUE` (`idCountry` ASC) VISIBLE,
  PRIMARY KEY (`Country_Name`),
  UNIQUE INDEX `Country_Name_UNIQUE` (`Country_Name` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4;

CREATE TABLE `test_db`.`city_info` (
  `City_name` VARCHAR(45) NOT NULL,
  `Station_Name` VARCHAR(100) NOT NULL,
  `State` VARCHAR(45) NOT NULL,
  `Country_Name` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`Station_Name`),
  UNIQUE INDEX `Station_Name_UNIQUE` (`Station_Name` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4;

ALTER TABLE `test_db`.`city_info` 

ADD INDEX `Country_Name_idx` (`Country_Name` ASC) VISIBLE;

ALTER TABLE `test_db`.`city_info` 
ADD CONSTRAINT `fkCountry_Name`
  FOREIGN KEY (`Country_Name`)
  REFERENCES `test_db`.`countries` (`Country_Name`)
  ON DELETE NO ACTION
  ON UPDATE NO ACTION;


  CREATE TABLE `test_db`.`air_quality_info` (
  `Station_Name` VARCHAR(100) NOT NULL,
  `Date` DATETIME NOT NULL,
  `PM2.5` TEXT NULL,
  `PM10` TEXT NULL,
  ` NO2` TEXT NULL,
  `NH3` TEXT NULL,
  `SO2` TEXT NULL,
  `CO` TEXT NULL,
  `OZONE` TEXT NULL,
  CONSTRAINT `Station_name`
    FOREIGN KEY (`Station_Name`)
    REFERENCES `test_db`.`city_info` (`Station_Name`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4;
