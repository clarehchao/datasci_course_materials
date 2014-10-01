.output similarity_matrix.txt

select sum(a.count*b.count)
from
(select * from Frequency where docid = '10080_txt_crude') as a
join
(select * from Frequency where docid = '17035_txt_earn') as b
on a.term = b.term;

.output keyword_search.txt
create view querydata as
select * from Frequency
union
select 'q' as docid, 'washington' as term, 1 as count
union
select 'q' as docid, 'taxes' as term, 1 as count
union
select 'q' as docid, 'treasury' as term, 1 as count;


select sum(a.count*b.count) as thesum from
(
(select * from querydata where docid = 'q') as a
join
(select * from Frequency) as b
on a.term = b.term
)
group by b.docid
order by thesum DESC LIMIT 1;

/* 'group by' statement should be outside of the ()
 before order by
 order by ... DEC => order the list in descedning order
 LIMIT x, retrieve the first xth row*/



