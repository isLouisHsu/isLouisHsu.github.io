-- https://zhuanlan.zhihu.com/p/52530057
-- https://blog.csdn.net/fashion2014/article/details/78826299

-------------------------------------------------------
-- 学生表
DROP TABLE student;
CREATE TABLE student (
    s_id varchar(20),               -- id
    s_name varchar(20) not null,    -- 姓名
    s_birth varchar(20) not null,   -- 生日
    s_sex VARCHAR(20) not null,     -- 性别
    PRIMARY KEY(s_id)
);

-- 教师表
DROP TABLE teacher;
CREATE TABLE teacher (
    t_id varchar(20),               -- id
    t_name varchar(20) not null,    -- 姓名
    PRIMARY KEY(t_id)
);

-- 课程表
DROP TABLE course;
CREATE TABLE course (
    c_id varchar(20),               -- id
    c_name varchar(20) not null,    -- 名称
    t_id varchar(20),               -- 教师id
    PRIMARY KEY(c_id)
);

-- 分数表
DROP TABLE score;
CREATE TABLE score (
    s_id varchar(20),               -- 学生id
    c_id varchar(20),               -- 课程id
    s_score INT(3),                 -- 课程分数
    PRIMARY KEY(s_id, c_id)
);

-------------------------------------------------------
insert into student 
(s_id, s_name, s_birth, s_sex) values
('1001' , 'student1' , '1990-01-01' , 'male'),
('1002' , 'student2' , '1990-12-21' , 'male'),
('1003' , 'student3' , '1990-05-20' , 'male'),
('1004' , 'student4' , '1990-08-06' , 'male'),
('1005' , 'student5' , '1991-12-01' , 'female'),
('1006' , 'student6' , '1992-03-01' , 'female'),
('1007' , 'student7' , '1989-07-01' , 'female'),
('1008' , 'student8' , '1990-01-20' , 'female');

insert into teacher
(t_id, t_name) values
('101' , 'teacher1'),
('102' , 'teacher2'),
('103' , 'teacher3');

insert into course 
(c_id, c_name, t_id) values
('01' , 'chinese' , '102'),
('02' , 'math' , '101'),
('03' , 'english' , '103');

insert into score
(s_id, c_id, s_score) values
('1001' , '01' , 80),
('1001' , '02' , 90),
('1001' , '03' , 99),
('1002' , '01' , 70),
('1002' , '02' , 60),
('1002' , '03' , 80),
('1003' , '01' , 80),
('1003' , '02' , 80),
('1003' , '03' , 80),
('1004' , '01' , 50),
('1004' , '02' , 30),
('1004' , '03' , 20),
('1005' , '01' , 76),
('1005' , '02' , 87),
('1006' , '01' , 31),
('1006' , '03' , 34),
('1007' , '02' , 89),
('1007' , '03' , 98);

-------------------------------------------------------
-- 1、查询"01"课程比"02"课程成绩高的学生的信息及课程分数
select st.*, sc.s_score_1, sc.s_score_2 from (
    -- s_id, s_score_1, s_score_2
    select s1.s_id, s1.s_score as s_score_1, s2.s_score as s_score_2 from score s1, score s2 
    where s1.c_id = '01' and s2.c_id = '02' and s1.s_id = s2.s_id and s1.s_score > s2.s_score
) sc left join student st on sc.s_id = st.s_id;

-- 2、查询"01"课程比"02"课程成绩低的学生的信息及课程分数
select st.*, sc.s_score_1, sc.s_score_2 from (
    -- s_id, s_score_1, s_score_2
    select s1.s_id, s1.s_score as s_score_1, s2.s_score as s_score_2 from score s1, score s2 
    where s1.c_id = '01' and s2.c_id = '02' and s1.s_id = s2.s_id and s1.s_score < s2.s_score
) sc left join student st on sc.s_id = st.s_id;

-- 3、查询平均成绩大于等于60分的同学的学生编号和学生姓名和平均成绩
select st.*, st_avg.avg_score from (
    -- s_id, avg_score
    select s_id, avg(s_score) as avg_score from score group by s_id having avg(s_score) >= 60
) st_avg left join student st on st_avg.s_id = st.s_id;

-- 4、查询平均成绩小于60分的同学的学生编号和学生姓名和平均成绩(包括有成绩的和无成绩的)
select st.*, st_avg.avg_score from (
    -- s_id, avg_score
    select s_id, avg(s_score) as avg_score from score group by s_id having avg(s_score) < 60
) st_avg left join student st on st_avg.s_id = st.s_id
union
-- 无成绩的，分数为0
select st.*, 0 as avg_score from student st 
where st.s_id not in (select distinct s_id from score);

-- 5、查询所有同学的学生编号、学生姓名、选课总数、所有课程的总成绩
select a.*, s.s_name from (
    select s_id, count(c_id), sum(s_score) from score group by s_id
) a left join student s on a.s_id = s.s_id;

-- 6、查询"李"姓老师的数量 
select count(*) from teacher where t_name like '李%';

-- 7、查询学过"张三"老师授课的同学的信息
select * from student where s_id in (       -- in
    -- the students who learn the courses;
    select distinct s_id from score where c_id in (
        -- the courses which taught by '张三'
        select distinct c.c_id from teacher t left join course c on t.t_id = c.t_id where t.t_name = '张三'
    )
);

-- 8、查询没学过"张三"老师授课的同学的信息 
select * from student where s_id not in (   -- not in
    -- the students who learn the courses;
    select distinct s_id from score where c_id in (
        -- the courses which taught by '张三'
        select distinct c.c_id from teacher t left join course c on t.t_id = c.t_id where t.t_name = '张三'
    )
);

-- 9、查询学过编号为"01"并且也学过编号为"02"的课程的同学的信息
select * from student where s_id in (
    select distinct s1.s_id from score s1, score s2 where s1.c_id = '01' and s2.c_id = '02'
);

-- 10、查询学过编号为"01"但是没有学过编号为"02"的课程的同学的信息
select * from student where s_id in (
    select s_id from score where c_id = '01' and s_id not in (
        select s_id from score where c_id = '02'
    )
);

-- 11、查询没有学全所有课程的同学的信息 
-- 12、查询至少有一门课与学号为"01"的同学所学相同的同学的信息 
-- 13、查询和"01"号的同学学习的课程完全相同的其他同学的信息 
-- 14、查询没学过"张三"老师讲授的任一门课程的学生姓名 
-- 15、查询两门及其以上不及格课程的同学的学号，姓名及其平均成绩 
-- 16、检索"01"课程分数小于60，按分数降序排列的学生信息
-- 17、按平均成绩从高到低显示所有学生的所有课程的成绩以及平均成绩
-- 18.查询各科成绩最高分、最低分和平均分：以如下形式显示：课程ID，课程name，最高分，最低分，平均分，及格率，中等率，优良率，优秀率
---- 及格为>=60，中等为：70-80，优良为：80-90，优秀为：>=90
-- 19、按各科成绩进行排序，并显示排名
-- 20、查询学生的总成绩并进行排名
-- 21、查询不同老师所教不同课程平均分从高到低显示 
-- 22、查询所有课程的成绩第2名到第3名的学生信息及该课程成绩
-- 23、统计各科成绩各分数段人数：课程编号,课程名称,[100-85],[85-70],[70-60],[0-60]及所占百分比
-- 24、查询学生平均成绩及其名次 
-- 25、查询各科成绩前三名的记录
---- 1.选出b表比a表成绩大的所有组
---- 2.选出比当前id成绩大的 小于三个的
-- 26、查询每门课程被选修的学生数 
-- 27、查询出只有两门课程的全部学生的学号和姓名 
-- 28、查询male生、female生人数 
-- 29、查询名字中含有"风"字的学生信息
-- 30、查询同名同性学生名单，并统计同名人数 
-- 31、查询1990年出生的学生名单
-- 32、查询每门课程的平均成绩，结果按平均成绩降序排列，平均成绩相同时，按课程编号升序排列 
-- 33、查询平均成绩大于等于85的所有学生的学号、姓名和平均成绩 
-- 34、查询课程名称为"数学"，且分数低于60的学生姓名和分数 
-- 35、查询所有学生的课程及分数情况； 
-- 36、查询任何一门课程成绩在70分以上的姓名、课程名称和分数； 
-- 37、查询不及格的课程
-- 38、查询课程编号为01且课程成绩在80分以上的学生的学号和姓名； 
-- 39、求每门课程的学生人数 
-- 40、查询选修"张三"老师所授课程的学生中，成绩最高的学生信息及其成绩
-- 41、查询不同课程成绩相同的学生的学生编号、课程编号、学生成绩 
-- 42、查询每门功成绩最好的前两名 
-- 43、统计每门课程的学生选修人数（超过5人的课程才统计）。要求输出课程号和选修人数，查询结果按人数降序排列，若人数相同，按课程号升序排列  
-- 44、检索至少选修两门课程的学生学号 
-- 45、查询选修了全部课程的学生信息 
-- 46、查询各学生的年龄
-- 47、查询本周过生日的学生
-- 48、查询下周过生日的学生
-- 49、查询本月过生日的学生
-- 50、查询下月过生日的学生
