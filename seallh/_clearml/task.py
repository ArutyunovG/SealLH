from __future__ import annotations
import logging

from clearml import Task
from omegaconf import DictConfig


logger = logging.getLogger("seallh._clearml.task")


class ClearMLTask:
    """Wrapper around clearml.Task that accepts clearml subconfig and handles setup."""

    def __init__(
        self,
        clearml_config: DictConfig
    ) -> None:
        """
        Initialize ClearML Task from clearml configuration.
        
        Args:
            clearml_config: The clearml section from the main config
        """
        # Extract basic task info
        project_name = clearml_config.get("project_name", "default_project")
        task_name = clearml_config.get("task_name", "default_task")
        output_uri = clearml_config.get("output_uri", None)
        
        logger.info("Initializing ClearML Task: %s/%s", project_name, task_name)
        
        # Prepare Task.init arguments
        task_args = {
            "project_name": project_name, 
            "task_name": task_name
        }
        
        # Only add output_uri if it's not None (null)
        if output_uri is not None:
            task_args["output_uri"] = output_uri
            if isinstance(output_uri, bool) and output_uri:
                logger.info("ClearML configured to store outputs on default remote server")
            elif isinstance(output_uri, str):
                logger.info("ClearML configured to store outputs at: %s", output_uri)
        
        # Initialize the task
        self.task = Task.init(**task_args)
        
        # Handle pip requirements for remote execution
        self._setup_pip_requirements(clearml_config)
        self._setup_docker_config(clearml_config)
        
        logger.info("ClearML Task initialized successfully. Task ID: %s", self.task.id)
    

    def _setup_pip_requirements(self, config: DictConfig) -> None:
        """Setup pip requirements from config."""
        pip_config = config.get("pip")
        if pip_config and "requirements" in pip_config:
            # Combine base requirements with project-specific extras
            base_requirements = list(pip_config.requirements)
            extra_requirements = list(pip_config.get("pip_extras", []))
            all_requirements = base_requirements + extra_requirements
            
            logger.info(f"Adding {len(base_requirements)} base + {len(extra_requirements)} extra pip requirements to task")
            logger.debug(f"Base requirements: {base_requirements}")
            if extra_requirements:
                logger.debug(f"Extra requirements: {extra_requirements}")
                
            self.task.set_packages(all_requirements)
        else:
            logger.debug("No pip requirements found in config")
    

    def _setup_docker_config(self, config: DictConfig) -> None:
        """Setup docker configuration from config."""
        docker_config = config.get("docker")
        if docker_config:
            image = docker_config.get("image")
            args = docker_config.get("args", [])
            base_env = docker_config.get("env", [])
            extra_env = docker_config.get("env_extras", [])
            
            logger.info("Setting up Docker configuration: %s", image)
     
            # Combine base and extra environment variables
            all_env = list(base_env) + list(extra_env)
            
            # Combine regular args with environment variable args
            combined_args = list(args) if args else []
            
            if all_env:
                # Add environment variables as -e arguments
                for env_var in all_env:
                    combined_args.extend(["-e", env_var])
                
                logger.info("Docker environment variables configured: %d base + %d extra", 
                           len(base_env), len(extra_env))
                logger.debug("All env vars: %s", [env_var.split("=")[0] for env_var in all_env])
            
            # Set docker configuration using the correct API
            self.task.set_base_docker(
                docker_image=image,
                docker_arguments=combined_args if combined_args else None
            )
            
            logger.info("Docker configuration applied successfully")
        else:
            logger.warning("No docker configuration found in config")

    # Convenience passthrough if needed later
    def __getattr__(self, item):
        return getattr(self.task, item)
