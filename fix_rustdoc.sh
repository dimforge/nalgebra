#! /bin/sh

echo "Integrating rust documentation to the website..."
cssfile=./docs/rustdoc/rustdoc.css
tmpcssfile=./docs/rustdoc/rustdoc.css.tmp

echo '@import url("//fonts.googleapis.com/css?family=Lato:400,700,900,400italic");' > $tmpcssfile
echo '@import url("//cdn.rawgit.com/piscis/github-fork-ribbon-css-bem/v0.1.22/dist/gh-fork-ribbon-bem.min.css");' >> $tmpcssfile
cat $cssfile >> $tmpcssfile
cat custom_flatly/css/bootstrap-custom.min2.css >> $tmpcssfile
cat custom_flatly/css/base2.css >> $tmpcssfile
cat custom_flatly/css/font-awesome.min.css >> $tmpcssfile
sed -i 's/margin-left: 230px;//g' $tmpcssfile
mv $tmpcssfile $cssfile

files=`find ./docs/rustdoc -name \*.html -printf '%p '`
sidebar='<nav class="sidebar">'
sub='<nav class="sub">'
container='<div class="container">'
class3='<div id="hide_medium" class="col-md-3">'
class9='<div class="col-md-9">'
end_div='</div>'
footer='<section class="footer"></section>'
body='<body[^>]*>'
head='<head>'
favicon='<link rel="shortcut icon" href="/img/favicon.ico">'


for file in `echo $files`
do
  echo "Patching $file."
  sed -i "s#$head#${head}${favicon}#g" $file
  sed -i "s#$sidebar#${container}${class3}${sidebar}#g" $file
  sed -i "s#$sub#${end_div}${class9}${sub}#g" $file
  sed -i "s#$footer#${end_div}${end_div}${footer}#g" $file
  sed -i "s#</body>\|</html>##g" $file
  sed -i "s#</head>#${css}</head>#g" $file

  nav='
  <div id='"'"'nav_placeholder'"'"'> </div>
  <script src="/jquery.js"></script>
  <script>
    var the_footer;
    $.get("/rustdoc_nalgebra/index.html", function(data) {
        data = data.split("../").join("/");
        data = data.split("..").join("/");
        var $data = $(data);
        $("div#nav_placeholder").prepend($data.find("#common_navbar"));
    });
  </script>'

  escaped_nav=`echo $nav | sed ':a;N;$!ba;s/\n/ /g'`
  sed -i "s%${body}%\0${escaped_nav}%g" $file
  fileend='
  <script>var base_url = "../" + window.rootPath;</script>
  <script src="/js/highlight.pack.js"></script>
  <script src="/js/bootstrap-3.0.3.min.js"></script>
  <script src="/js/base.js"></script>

</body>
</html>'
  echo $fileend >> $file
done
echo "... integration done!"
