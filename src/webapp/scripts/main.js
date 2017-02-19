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
                $('#styleguide').text(response.style);
                $('#textresponse').text(response.anonymized);
                $('#msg').text(response.msg);
            }
        });
    });
});
