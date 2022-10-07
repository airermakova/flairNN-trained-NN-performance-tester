import os
import web
 

urls = ('/', 'Upload')

class Upload:
    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return """<html><head></head><body>


<form method="POST" enctype="multipart/form-data" action="">
<input type="file" name="myfile" />
<br/>
<br/>
First phrases to take (in case you want to take phrase from the beginning leave blanc): 
<input type="number" name="phSt"/>
<br/>
<br/>
Number of phrases to take (in case you want to take all phrases leave blank): 
<input type="number" name="phNum" text "Submit"/>
<br/>
<br/>
<input type="submit" />
</form>
</body></html>"""
    def POST(self):
        x = web.input(myfile={})
        phS = 0
        phN = 0
        phSStr = web.input(phSt={})
        phNStr = web.input(phNum={})
        print("START PHRASE - " + str(x['phSt']))
        print("FINISH PHRASE - " + str(x['phNum']))
        if len(x['phSt'])>0:
            phS = int(x['phSt'])
        if len(x['phNum'])>0:
            phN = int(x['phNum'])
        filedir = os.getcwd() # change this to the directory you want to store the file in.
        if 'myfile' in x: # to check if the file-object is created
            filepath=x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
            filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
            fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
            fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.
        raise web.seeother('/')


if __name__ == "__main__":
   app = web.application(urls, globals()) 
   app.run()