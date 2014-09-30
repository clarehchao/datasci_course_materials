.output select.txt
select count(*) from Frequency where docid = '10398_txt_earn';

.output select_project.txt
select count(distinct term) from Frequency where docid = '10398_txt_earn' and count=1;

.output union.txt
select count(*) from
(
select term from Frequency where docid = '10398_txt_earn' and count=1
union
select term from Frequency where docid = '925_txt_trade' and count=1
) x;

.output count.txt
select count(count) from Frequency where term = 'parliament';

.output big_documents.txt
select count(*) from
(
select docid,sum(count) as countsum from Frequency group by docid
having countsum > 300
) x;

.output two_words.txt
select count(*) from
(
select docid from Frequency where term = 'transactions'
intersect
select docid from Frequency where term = 'world'
) x;

.output two_words2.txt
select count(*) from
(
(select docid from Frequency where term = 'transactions') as t1
join (select docid from Frequency where term = 'world') as t2
on t1.docid = t2.docid
) x;
