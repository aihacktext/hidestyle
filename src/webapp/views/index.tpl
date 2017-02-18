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
    <div>
      <form action="/post" method="post">
        % if anonymized:
        <div class="hint">{{anonymized}}</div>
        % end
        % if msg:
        <div class="hint">{{msg}}</div>
        % end
        <input type="input" name="text" value="{{orig_text}}"><br>
        <input type="submit" value="Submit">
      </form>
    </div>
  </body>
</html>
