<html>
  <head>
    <title>Text anonymizer</title>
    <style>
      html {
        background-color: #fffefa;
        color: #303030;
      }
      #header {
        text-align: center;
      }
      form {
        border: 1px solid #E4CB8B;
        padding: 2em;
        margin: 4em;
        border-radius: 1em;
      }
      div.hint {
        padding: .2em;
      }
    </style>
  </head>
  <body>
   <img src="http://txt-dynamic.static.1001fonts.net/txt/dHRmLjcyLjAwMDAwMC5RVzV2Ym5sdGFYTmxjZywsLjE,/got-heroin.regular.png">
    <div>
      <form action="/post" method="post">
        <input type="input" name="text" value="{{orig_text}}"><br>
        <input type="submit" value="Submit">
         % if anonymized:
        <div class="hint">{{anonymized}}</div>
        % end
        % if msg:
        <div class="hint">{{msg}}</div>
        % end
     </form>
    </div>
  </body>
</html>
