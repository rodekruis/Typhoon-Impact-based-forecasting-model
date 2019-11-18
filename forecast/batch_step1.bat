wget --no-check-certificate --keep-session-cookies --save-cookies tsr_cookies.txt --post-data "user=RodeKruis&pass=TestRK1" -O loginresult.txt "https://www.tropicalstormrisk.com/business/checkclientlogin.php?script=true"
wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O RodeKruis.xml "https://www.tropicalstormrisk.com/business/include/dlxml.php?f=RodeKruis.xml"
