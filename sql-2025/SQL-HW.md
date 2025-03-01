SQL HW



```sql

-- CREATE DATABASE IF NOT EXISTS SchoolDB;
-- USE SchoolDB;

DROP TABLE IF EXISTS Customers; -- , Orders, Shippings;
DROP TABLE IF EXISTS Shippings; -- , Orders, Shippings;
DROP TABLE IF EXISTS Orders; -- , Orders, Shippings;
DROP TABLE IF EXISTS Enrollment;
DROP TABLE IF EXISTS Students;
DROP TABLE IF EXISTS Courses;
DROP TABLE IF EXISTS Enrollment;

 
CREATE TABLE Students (
student_id  SERIAL PRIMARY KEY  ,
name VARCHAR(50) NOT NULL,
age INT,
grade VARCHAR(10)
);

CREATE TABLE Courses (
course_id  SERIAL PRIMARY KEY  ,
course_name VARCHAR(100) NOT NULL,
instructor VARCHAR(50)
) ;

INSERT INTO Students (name, age, grade) VALUES
('Alice', 20, 'A'),
('Bob', 22, 'B'),
('Charlie', 21, 'A'),
('Diana', 23, 'C'),
('Eve', 19, 'B');


INSERT INTO Courses (course_name, instructor) VALUES
('Mathematics', 'Dr. Smith'),
('Computer Science', 'Prof. Johnson'),
('Physics', 'Dr. Lee');


-- 6. Retrieve All Students
SELECT * FROM Students;

-- 7. Retrieve Students Who Scored 'A'
SELECT * FROM Students WHERE grade = 'A';

-- 8. Find Students Older than 21
SELECT * FROM Students WHERE age > 21;

-- 9. Order Students by Age
SELECT * FROM Students ORDER BY age DESC;

-- 10. Retrieve Distinct Grades
-- Task: Find all unique grades in the Students table.
SELECT DISTINCT grade FROM Students;


-- Step 4: Updating and Deleting Data

-- 11. Update a Student's Grade. Task: Update Bob's grade to ' A '.
UPDATE Students SET grade = 'A' WHERE name = 'Bob';


-- 12. Delete a Student. Task: Remove Diana from the Students table.
DELETE FROM Students WHERE name = 'Diana';


-- Step 5: Advanced Queries
-- 13. Count the Number of Students
SELECT COUNT(*) FROM Students;

-- 14. Find the Average Age of Students
SELECT AVG(age) AS average_age FROM Students;

-- 15. Find the Youngest Student
SELECT * FROM Students WHERE age = (SELECT MIN(age) FROM Students);

-- 16. Group Students by Grade
SELECT grade, COUNT(*) FROM Students GROUP BY grade;

-- 17. Find Students with Names Starting with ' A^'
SELECT * FROM Students WHERE name LIKE 'A%';

-- 18. Find Students Aged Between 20 and 22
SELECT * FROM Students WHERE age BETWEEN 20 AND 22;

-- 19. Find Students in a Specific Set of Grades\
SELECT * FROM Students WHERE grade IN ('A', 'B');



-- Step 6: Joins

-- 20. Create an Enrollment Table

CREATE TABLE Enrollment (
    enrollment_id SERIAL PRIMARY KEY  ,
    student_id INT,
    course_id INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

-- 21. Insert Data into Enrollment Table
INSERT INTO Students (student_id, name, age, grade) VALUES
(4, 'Diana', 23, 'C');
INSERT INTO Enrollment (student_id, course_id) VALUES
(1, 1),
(1, 2),
(2, 2),
(3, 1),
(4, 3);

-- 22. Join Students and Enrollment Tables
SELECT Students.name, Courses.course_name
FROM Students
JOIN Enrollment ON Students.student_id = Enrollment.student_id
JOIN Courses ON Enrollment.course_id = Courses.course_id;


-- 23. Left Join Students with Enrollment
SELECT Students.name, Courses.course_name
FROM Students
LEFT JOIN Enrollment ON Students.student_id = Enrollment.student_id
LEFT JOIN Courses ON Enrollment.course_id = Courses.course_id;

-- Step 7: Indexing & Optimization
-- 24. Create an Index on Name
CREATE INDEX idx_name ON Students(name);



-- 2. Verify Index Exists
SELECT indexname, tablename FROM pg_indexes WHERE tablename = 'students';

 

-- 25. Drop the Index on Name
DROP INDEX idx_name; -- DROP INDEX idx_name ON Students;

-- 26. Show Indexes on Students Table
-- SHOW INDEXES FROM Students;
SELECT indexname, tablename, indexdef 
FROM pg_indexes 
WHERE tablename = 'students';



-- Step 8: Query Optimization
-- 27. Limit the Number of Results
SELECT * FROM Students LIMIT 3;


-- 28. Use Aliases for Readability
SELECT name AS StudentName FROM Students;

-- 29. Find Students Without Enrollment
SELECT name FROM Students WHERE student_id NOT IN (SELECT student_id FROM Enrollment);


-- Step 9: Table Modifications
-- 30. Add a Column to the Students Table
ALTER TABLE Students ADD COLUMN email VARCHAR(100);


-- 31. Drop a Column from the Students Table
ALTER TABLE Students DROP COLUMN email;


-- Step 10: Cleanup
-- 32. Delete All Students
DELETE FROM Students;

-- 33. Drop the Enrollment Table
DROP TABLE Enrollment;

-- 34. Drop the Courses Table
DROP TABLE Courses;

-- 35. Drop the SchooIDB Database
-- DROP DATABASE SchooldB;




```

