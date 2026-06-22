import os
import git
from conf.paths import repo_path
# List of repositories (directory paths) to monitor
repository_paths = [rf"{repo_path}\pr", rf"{repo_path}\sleep_pose", rf"{repo_path}\sleep_cycle"]

# Function to check for changes in a repository and update the local main branch
def git_files(repository_paths):
    def check_and_update_repository(repo_path):
        try:
            repo = git.Repo(repo_path)
            remote_name = "origin"  # The default name for the remote repository

            # Fetch the latest changes from the remote repository
            repo.remotes[remote_name].fetch()

            # Get the current commit on the main branch
            current_commit = repo.heads.main.commit

            # Check if the remote branch has new commits
            remote_head = repo.remotes[remote_name].refs["main"].commit
            if current_commit != remote_head:
                print(f"Changes detected in '{repo_path}' repository.")
                print("Updating local main branch...")
                # Fetch and update the local main branch
                repo.git.pull(remote_name, "main")

                # Reset the local main branch to the remote head
                repo.heads.main.set_commit(remote_head)
                repo.heads.main.checkout()
                print("Local main branch updated successfully.")
                
                return repo_path
            else:
                print(f"No changes detected in '{repo_path}' repository.")
                return None  # Return None when no changes are detected

        except git.exc.NoSuchPathError:
            print(f"The repository '{repo_path}' does not exist locally.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None  # Return None when an error occurs

    changes = []  # Create a list to store repository paths with changes
    # Loop through the list of repositories and check for updates
    for repo_path in repository_paths:
        change = check_and_update_repository(repo_path)
        if change is not None:  # Only add the path to the list if a change was detected
            changes.append(change)
    return changes  # Return the list of repository paths with changes