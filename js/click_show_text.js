var a_idx = 0;
jQuery(document).ready(function($) {
    $("body").click(function(e) {
        var a = new Array
        ("for", "while", "catch", "except", "if", "range", 
        "class", "min", "max", "sort", "map", "filter", 
        "lambda", "switch", "case", "iter", "next", "enum", "struct",  
        "void", "int", "float", "double", "char", "signed", "unsigned");
        var $i = $("<span/>").text(a[a_idx]);
        a_idx = (a_idx + 3) % a.length;
        var x = e.pageX, 
        y = e.pageY;
        $i.css({
            "z-index": 5,
            "top": y - 20,
            "left": x,
            "position": "absolute",
            "font-weight": "bold",
            "color": "#333333"
        });
        $("body").append($i);
        $i.animate({
            "top": y - 180,
            "opacity": 0
        },
			3000,
			function() {
			    $i.remove();
			});
    });
    setTimeout('delay()', 2000);
});

function delay() {
    $(".buryit").removeAttr("onclick");
}