import { LucideIcon } from 'lucide-react';
import { Button } from './button';
import { cn } from '@/lib/utils';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className
}: EmptyStateProps) {
  return (
    <div className={cn(
      'flex flex-col items-center justify-center py-12 sm:py-16 md:py-20 px-4 text-center',
      className
    )}>
      {Icon && (
        <div className="mb-4 p-4 rounded-full bg-muted/30">
          <Icon className="h-8 w-8 sm:h-10 sm:w-10 md:h-12 md:w-12 text-muted-foreground" />
        </div>
      )}
      <h3 className="text-lg sm:text-xl md:text-2xl font-semibold text-foreground mb-2">
        {title}
      </h3>
      {description && (
        <p className="text-sm sm:text-base text-muted-foreground max-w-md mb-6">
          {description}
        </p>
      )}
      {action && (
        <Button onClick={action.onClick} size="lg">
          {action.label}
        </Button>
      )}
    </div>
  );
}
