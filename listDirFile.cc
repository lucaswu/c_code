#include<vector>
#include<stdio.h>
#include<string>
#include<dirent.h> 
#include<memory.h> 
#include <algorithm>
using namespace std;
void listDir(const char *path,vector<string>& files)
{
	DIR *pDir;
	struct dirent *ent;
	char childpath[512];
	char absolutepath[512];
	pDir = opendir(path);
	memset(childpath, 0, sizeof(childpath));
	while ((ent = readdir(pDir)) != NULL)
	{
		if (ent->d_type & DT_DIR)
		{
 
			if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
			{
				continue;
			}
			sprintf(childpath, "%s/%s", path, ent->d_name);
			listDir(childpath,files);
			
		}
		else
		{
			sprintf(absolutepath, "%s/%s", path, ent->d_name);
			files.push_back(absolutepath);
		}
	}
 
	sort(files.begin(),files.end());
 
}


int main(int argc,char*argv[])
{
	std::vector<string> files;
	listDir(argv[1],files);
	for(int i=0;i<files.size();i++){
		printf("%d=%s\n",i,files[i].c_str());
	}
}