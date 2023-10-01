--Qurey 1
select "Marital Status", AVG(Age) AS "Average Age"
from customer
group by "Marital Status"

--Query 2
select
    case
        when "gender"::integer = 1 then 'Pria'
        when "gender"::integer = 0 then 'Wanita'
        else 'Tidak Diketahui'
    end as "Gender",
    AVG(Age) as "Average Age"
from customer
group by "gender"

--Query 3
select storename as "Store Name", SUM(qty) as "Total Quantity"
from transaction
join store on transaction.storeid = store.storeid
group by storename
order by "Total Quantity" desc
limit 1

--Query 4
select "Product Name", SUM("totalamount") AS "Total Amount"
from transaction
join product on product.productid = transaction.productid
group by "Product Name"
order by "Total Amount" DESC
limit 1