"""
Repository management utilities for cloning and installing external dependencies.
"""
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional
from omegaconf import DictConfig


logger = logging.getLogger("seallh.experiment.external_repository_setuper")


class ExternalRepositorySetuper:

    """
    Sets up external repository dependencies.
    Handles cloning, checkout, and installation of external repos.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize repository manager.
        
        Args:
            config: Configuration containing repository specifications
            repos_dir: Optional explicit directory where repositories will be cloned.
        """
        self.config = config

        repos_dir = self.config.get("external_repos_dir", None)
        if not repos_dir:
            repos_dir = './external_repos'

        self.repos_dir = Path(repos_dir).expanduser().resolve()
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Repository manager initialized with repos_dir: {self.repos_dir}")
    

    def setup_repositories(self) -> Dict[str, str]:
        """
        Set up all configured repositories.
        
        Returns:
            Dictionary mapping repository names to their local paths
        """
        repo_paths = {}
        
        repositories = self.config.get("external_repos", [])
        
        if not repositories:
            logger.info("No external repositories configured")
            return repo_paths
        
        logger.info(f"Setting up {len(repositories)} external repositories...")
        
        for repo_config in repositories:
            try:
                repo_name, repo_path = self._setup_single_repository(repo_config)
                repo_paths[repo_name] = str(repo_path)
                logger.info(f"Repository '{repo_name}' ready at: {repo_path}")
            except Exception as e:
                logger.error(f"Failed to setup repository '{repo_name}': {e}")
                raise
        
        return repo_paths


    def _setup_single_repository(self, repo_config: DictConfig) -> Path:
        """
        Set up a single repository.
        
        Args:
            repo_name: Name of the repository
            repo_config: Repository configuration
            
        Returns:
            Tuple of name and local path of the repository
        """
        repo_name = repo_config.get("name")
        assert repo_name, "Repository entry missing 'name' field"
        repo_url = repo_config.get("url")
        assert repo_url, f"Repository '{repo_name}' missing 'url' field"
        url = repo_config.get("url")

        branch = repo_config.get("branch", None)
        tag = repo_config.get("tag", None)
        if branch or tag:
            assert bool(branch) ^ bool(tag), "Cannot specify both branch and tag for cloning"

        install_method = repo_config.get("install_method", None)
        
        repo_path = self.repos_dir / repo_name
        
        # Clone or update repository
        if repo_path.exists():
            logger.info(f"Repository '{repo_name}' already exists, updating...")
            self._update_repository(repo_path, branch, tag)
        else:
            logger.info(f"Cloning repository '{repo_name}' from {url}")
            self._clone_repository(url, repo_path, branch, tag)
        
        # Install repository
        if install_method:
            self._install_repository(repo_path, install_method)
        
        return (repo_name, repo_path)


    def _clone_repository(self, url: str, repo_path: Path, branch: str, tag: Optional[str]):
        """Clone a repository."""
        clone_cmd = ["git", "clone", url, str(repo_path)]
        
        if branch:
            clone_cmd.extend(["--branch", branch])
        
        logger.info(f"Running: {' '.join(clone_cmd)}")
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")
        
        # Checkout specific tag if specified
        if tag:
            self._checkout_tag(repo_path, tag)
        
        # Initialize and update submodules
        self._update_submodules(repo_path)


    def _update_repository(self, repo_path: Path, branch: str, tag: Optional[str]):
        """Update an existing repository."""
        # Fetch latest changes
        fetch_cmd = ["git", "fetch", "--all"]
        logger.info(f"Running: {' '.join(fetch_cmd)} in {repo_path}")
        result = subprocess.run(fetch_cmd, cwd=repo_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Failed to fetch updates: {result.stderr}")
        
        # Checkout branch or tag
        if tag:
            self._checkout_tag(repo_path, tag)
        elif branch:
            checkout_cmd = ["git", "checkout", branch]
            logger.info(f"Running: {' '.join(checkout_cmd)} in {repo_path}")
            result = subprocess.run(checkout_cmd, cwd=repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Failed to checkout branch '{branch}': {result.stderr}")
            
            # Pull latest changes
            pull_cmd = ["git", "pull", "origin", branch]
            logger.info(f"Running: {' '.join(pull_cmd)} in {repo_path}")
            result = subprocess.run(pull_cmd, cwd=repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Failed to pull latest changes: {result.stderr}")
        
        # Update submodules
        self._update_submodules(repo_path)


    def _checkout_tag(self, repo_path: Path, tag: str):
        """Checkout a specific tag."""
        checkout_cmd = ["git", "checkout", tag]
        logger.info(f"Running: {' '.join(checkout_cmd)} in {repo_path}")
        result = subprocess.run(checkout_cmd, cwd=repo_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout tag '{tag}': {result.stderr}")


    def _update_submodules(self, repo_path: Path):
        """Initialize and update git submodules."""
        # Check if .gitmodules file exists
        gitmodules_path = repo_path / ".gitmodules"
        if (not gitmodules_path.exists()) or (not gitmodules_path.is_file()):
            logger.info(f"No .gitmodules file found in {repo_path}, skipping submodule update")
            return
        
        logger.info(f"Updating submodules in {repo_path}")
        
        # Initialize and update submodules recursively in one command
        update_cmd = ["git", "submodule", "update", "--init", "--recursive"]
        logger.info(f"Running: {' '.join(update_cmd)} in {repo_path}")
        result = subprocess.run(update_cmd, cwd=repo_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Failed to update submodules: {result.stderr}")
        else:
            logger.info("Submodules updated successfully")


    def _install_repository(self, repo_path: Path, install_method: str):
        """Install a repository using the specified method."""
        if install_method == "pip_editable":
            self._pip_install_editable(repo_path)
        elif install_method == "pip":
            self._pip_install(repo_path)
        elif install_method == "none":
            logger.info("Skipping installation (install_method: none)")
        else:
            logger.warning(f"Unknown install method: {install_method}")
    
    def _pip_install_editable(self, repo_path: Path):
        """Install repository in editable mode using pip."""
        install_cmd = ["pip", "install", "-e", str(repo_path)]
        logger.info(f"Running: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install repository in editable mode: {result.stderr}")
        
        logger.info("Repository installed in editable mode successfully")
    
    def _pip_install(self, repo_path: Path):
        """Install repository using pip."""
        install_cmd = ["pip", "install", str(repo_path)]
        logger.info(f"Running: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install repository: {result.stderr}")
        
        logger.info("Repository installed successfully")


def setup_repositories(config: DictConfig) -> Dict[str, str]:
    """
    Convenience function to set up repositories.
    
    Args:
        config: Configuration containing repository specifications
        
    Returns:
        Dictionary mapping repository names to their local paths
    """
    manager = ExternalRepositorySetuper(config)
    return manager.setup_repositories()
