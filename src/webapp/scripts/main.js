$(document).ready(function(){
    $('#quotearea').hide();

    $("#anonymizerform").submit(function(e){
        e.preventDefault(e);
        var form = $(this);
        $.ajax({ 
            url   : '/anonymize',
            data: form.serialize(),
            crossDomain: true,
            success: function(response){
                $('#quotearea').show();
                $('#textresponse').text(response.anonymized);
                $('#msg').text(response.msg);
                $('#table').bootstrapTable({
                     columns: [{
                        field: 'metric',
                        title: 'Metric Name'
                    }, {
                        field: 'score',
                        title: 'Score'
                    }],
                    data: response.style
                });

            }
        });
    });
});
