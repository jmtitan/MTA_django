//
$('#myTabs a').click(function (e) {
    e.preventDefault()
    $(this).tab('show')
});

$(document).ready(function() {



    Highcharts.setOptions({
        colors: ['#8129dd', '#8ec63f','#2756ca','#f1b601','#f86423','#27aae3']
    });


    var chart8 = $("#fbt-monthchart").highcharts({
        chart: {
            backgroundColor: '#1e2131',
            plotBackgroundColor: null,
            plotBorderWidth: null,
            plotShadow: false
        },
        title: false,
        tooltip: {
            headerFormat: '{series.name}<br>',
            pointFormat: '{point.name}: <b>{point.percentage:.1f}%</b>'
        },
        exporting:{ enabled:false, //用来设置是否显示‘打印’,'导出'等功能按钮，不设置时默认为显示
        },
        credits:{
            enabled: false // 禁用版权信息
        },
        legend: {
            layout: 'horizontal',
            align: 'center',
            verticalAlign: 'bottom',
            itemStyle: { cursor: 'pointer', color: 								'#FFF' },
            itemHiddenStyle: { color: '#CCC' },
        },
        plotOptions: {
            pie: {
                allowPointSelect: true,
                cursor: 'pointer',
                dataLabels: {
                    enabled: true,
                    format: '<b>{point.name}</b>: {point.percentage:.1f} %',
                    style: {
                        color: (Highcharts.theme && Highcharts.theme.contrastTextColor) || '#FFF'
                    }
                }
            }
        },
        series: [{
            type: 'pie',
            name: '各广告归因占比',
            data: [
                {
                    name: '广告A',
                    y: 10,
                    sliced: true,
                    selected: true
                },
                ['广告B',   15.0],
                ['广告C',       20.0],
                ['广告D',   15.0],
                ['广告E',   5.0],
                ['广告F',   35.0],
            ]
        }]
    });



    var chart10 = $("#zxlxchart").highcharts({
        chart: {
            backgroundColor: '#1e2131',
            type: 'column',
            plotBorderColor: '#1c2a38',
            plotBorderWidth: 1,
        },
        title:false,
        xAxis: {
            gridLineColor: '#1c2a38',//网格线
            tickColor:'#1c2a38',//刻度线
            lineColor: '#1c2a38',//轴线
            categories: ['广告A', '广告B', '广告C', '广告D', '广告E','广告F']
        },
        yAxis: {
            min: 0,
            title:false,
            gridLineColor: '#1c2a38',//网格线
            tickColor:'#1c2a38',//刻度线
            stackLabels: {
                enabled: true,
                style: {
                    fontWeight: 'bold',
                    color: (Highcharts.theme && Highcharts.theme.textColor) || '#fff'
                }//柱形图上方数据显示
            }
        },
        exporting:{ enabled:false, //用来设置是否显示‘打印’,'导出'等功能按钮，不设置时默认为显示
        },
        credits:{
            enabled: false // 禁用版权信息
        },
        legend: {
            layout: 'horizontal',
            align: 'center',
            verticalAlign: 'bottom',
            itemStyle: { cursor: 'pointer', color: 								'#FFF' },
            itemHiddenStyle: { color: '#CCC' },
        },
        tooltip: {
            formatter: function () {
                return '<b>' + this.x + '</b><br/>' +
                    this.series.name + ': ' + this.y + '<br/>' +
                    '总量: ' + this.point.stackTotal;
            }
        },
        plotOptions: {
            column: {           //不显示阴影
                stacking: 'normal',
                bar: false,
                borderWidth: 0,  //柱子边框的大小
            },
        },
        series: [{
            name: '百万元',
            data: [15, 3, 24, 7, 2,4]
        }]
    });


});