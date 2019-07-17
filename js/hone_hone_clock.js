!function (e, t, a) { 
	/* code */
	var initClock = function(){
	  var sHtml = '';
	  sHtml += '<div style="position: fixed;right: 10px;top: 80px;width: 160px;height: 70px;">';
	  sHtml += '  <object classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" codebase="http://fpdownload.macromedia.com/pub/shockwave/cabs/flash/swflash.cab#version=8,0,0,0" width="160" height="70" id="honehoneclock" align="middle">';
	  sHtml += '    <param name="allowScriptAccess" value="always">';
	  sHtml += '    <param name="movie" value="http://chabudai.sakura.ne.jp/blogparts/honehoneclock/honehone_clock_tr.swf">';
	  sHtml += '    <param name="quality" value="high">';
	  sHtml += '    <param name="bgcolor" value="#ffffff">';
	  sHtml += '    <param name="wmode" value="transparent">';
	  sHtml += '    <embed wmode="transparent" src="http://chabudai.sakura.ne.jp/blogparts/honehoneclock/honehone_clock_tr.swf" quality="high" bgcolor="#ffffff" width="160" height="70" name="honehoneclock" align="middle" allowscriptaccess="always" type="application/x-shockwave-flash" pluginspage="http://www.macromedia.com/go/getflashplayer">';
	  sHtml += '    </object>';
	  sHtml += '</div>';
	  
	  t = t || document;    
	  t.write(sHtml);
	}
	initClock();
  }(window, document);